"""Audio restoration and source separation."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

STEM_NAMES = ("vocals", "drums", "bass", "other")


class AudioError(RuntimeError):
    """Something went wrong that the user should see in plain language."""


@dataclass
class Probe:
    duration: float | None
    sample_rate: int | None
    highest_frequency: int | None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def separation_backend() -> str:
    return os.getenv("SEPARATION_BACKEND", "replicate").lower()


async def _run(*args: str, timeout: int = 1800) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise AudioError("Processing took too long and was stopped. Try a shorter track.")
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip().splitlines()[-4:]
        raise AudioError("Audio processing failed: " + " ".join(tail))
    return out.decode(errors="replace")


async def probe(path: Path) -> Probe:
    """Read duration and sample rate, and estimate where the real content stops.

    The highest frequency matters: a track that dies at 15 kHz was cut from a
    lossy source or an old tape, and no amount of processing puts that back.
    """
    if not ffmpeg_available():
        raise AudioError("ffmpeg isn't installed on the server.")

    raw = await _run(
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path), timeout=60,
    )
    data = json.loads(raw or "{}")
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    duration = float(data.get("format", {}).get("duration", 0)) or None
    sample_rate = int(stream.get("sample_rate", 0)) or None

    highest = None
    if sample_rate:
        # Rough cutoff estimate: bitrate is a decent proxy for lossy lowpass.
        bitrate = int(stream.get("bit_rate") or data.get("format", {}).get("bit_rate") or 0)
        codec = (stream.get("codec_name") or "").lower()
        if codec in {"flac", "alac", "pcm_s16le", "pcm_s24le", "wavpack"}:
            highest = sample_rate // 2
        elif bitrate:
            kbps = bitrate // 1000
            highest = 15000 if kbps < 128 else 16000 if kbps < 192 else 19000 if kbps < 256 else 20000
    return Probe(duration=duration, sample_rate=sample_rate, highest_frequency=highest)


def _restore_filters(*, hum_hz: int, denoise: float, presence: float) -> str:
    """Build the ffmpeg filter chain.

    Ordered the way a restoration engineer would: kill rumble, kill hum, kill
    broadband hiss, repair transient damage, then shape tone, and only then
    touch level.
    """
    denoise = max(0.0, min(1.0, denoise))
    presence = max(0.0, min(1.0, presence))

    chain = ["highpass=f=32:p=2"]  # turntable rumble and tape wow

    if hum_hz in (50, 60):
        # Mains hum sits on the fundamental and its harmonics.
        for h in range(1, 5):
            f = hum_hz * h
            chain.append(f"equalizer=f={f}:t=q:w=25:g=-18")

    if denoise > 0:
        # afftdn does spectral subtraction — this is what kills tape hiss.
        nr = round(6 + denoise * 20, 1)      # 6..26 dB of reduction
        nf = round(-45 + denoise * 20, 1)    # noise floor estimate
        chain.append(f"afftdn=nr={nr}:nf={nf}:tn=1")

    # Vinyl clicks, then clipping repair — a lot of 60s masters were driven hard.
    chain.append("adeclick=w=55:o=75:a=2:t=2:b=2:m=a")
    chain.append("adeclip=w=55:o=75:a=8:t=10:m=a")

    if presence > 0:
        # A gentle presence lift and a touch of air. Deliberately modest — an
        # aggressive top-end boost on a 1965 master just amplifies hiss.
        chain.append(f"equalizer=f=3200:t=q:w=1.4:g={round(presence * 3.5, 2)}")
        chain.append(f"equalizer=f=9500:t=q:w=0.9:g={round(presence * 2.5, 2)}")
        chain.append(f"equalizer=f=110:t=q:w=1.1:g={round(presence * 1.5, 2)}")

    chain.append("aresample=44100:resampler=soxr:precision=28")
    return ",".join(chain)


async def restore(src: Path, dst: Path, *, hum_hz: int, denoise: float, presence: float) -> Path:
    filters = _restore_filters(hum_hz=hum_hz, denoise=denoise, presence=presence)
    await _run(
        "ffmpeg", "-y", "-i", str(src),
        "-af", filters,
        "-ac", "2", "-c:a", "pcm_s16le", str(dst),
    )
    return dst


async def normalise(src: Path, dst: Path) -> Path:
    """Two-pass EBU R128 loudness normalisation to streaming level."""
    measured = await _run(
        "ffmpeg", "-i", str(src),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    )
    await _run(
        "ffmpeg", "-y", "-i", str(src),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-ar", "44100", "-c:a", "pcm_s16le", str(dst),
    )
    del measured
    return dst


async def mixdown(inputs: list[tuple[Path, float]], dst: Path) -> Path:
    """Sum stems at the given gains, with a limiter so a hot balance can't clip."""
    args: list[str] = ["ffmpeg", "-y"]
    for path, _ in inputs:
        args += ["-i", str(path)]

    parts, labels = [], []
    for i, (_, gain) in enumerate(inputs):
        g = max(0.0, min(4.0, gain))
        parts.append(f"[{i}:a]volume={g:.3f}[g{i}]")
        labels.append(f"[g{i}]")

    parts.append(f"{''.join(labels)}amix=inputs={len(inputs)}:normalize=0[sum]")
    parts.append("[sum]alimiter=limit=0.89:level=disabled[out]")

    args += ["-filter_complex", ";".join(parts), "-map", "[out]"]
    if dst.suffix == ".mp3":
        args += ["-c:a", "libmp3lame", "-b:a", "320k"]
    else:
        args += ["-c:a", "pcm_s16le"]
    args.append(str(dst))

    await _run(*args, timeout=900)
    return dst


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------

async def separate(src: Path, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    backend = separation_backend()
    if backend == "replicate":
        return await _separate_replicate(src, outdir)
    if backend == "local":
        return await _separate_local(src, outdir)
    raise AudioError(f"Unknown SEPARATION_BACKEND '{backend}'. Use 'replicate' or 'local'.")


async def _separate_local(src: Path, outdir: Path) -> dict[str, Path]:
    """Run Demucs on this machine. Free, but slow without a GPU."""
    model = os.getenv("DEMUCS_MODEL", "htdemucs")
    await _run(
        "python", "-m", "demucs",
        "-n", model, "-o", str(outdir), "--out-ext", "wav",
        str(src), timeout=3600,
    )
    root = outdir / model / src.stem
    stems = {}
    for name in STEM_NAMES:
        path = root / f"{name}.wav"
        if path.is_file():
            final = outdir / f"{name}.wav"
            shutil.move(str(path), final)
            stems[name] = final
    if not stems:
        raise AudioError("Demucs ran but produced no stems.")
    return stems


async def _separate_replicate(src: Path, outdir: Path) -> dict[str, Path]:
    """Run Demucs on Replicate. Needs REPLICATE_API_TOKEN."""
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise AudioError(
            "REPLICATE_API_TOKEN isn't set on the server, so separation can't run. "
            "Add it to .env, or set SEPARATION_BACKEND=local."
        )

    try:
        import replicate
    except ImportError as exc:
        raise AudioError("The replicate package isn't installed.") from exc

    model = os.getenv("REPLICATE_DEMUCS_MODEL", "ryan5453/demucs")

    def _call():
        with src.open("rb") as fh:
            return replicate.run(
                model,
                input={
                    "audio": fh,
                    "stem": "none",           # return all four stems
                    "output_format": "wav",
                    "model_name": os.getenv("DEMUCS_MODEL", "htdemucs"),
                },
            )

    try:
        output = await asyncio.to_thread(_call)
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Replicate rejected the job: {exc}") from exc

    urls = _extract_urls(output)
    if not urls:
        raise AudioError("Replicate returned no stems.")

    stems: dict[str, Path] = {}
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        for name, url in urls.items():
            dest = outdir / f"{name}.wav"
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            stems[name] = dest
    return stems


def _extract_urls(output) -> dict[str, str]:
    """Replicate models are inconsistent about output shape. Handle both."""
    if isinstance(output, dict):
        return {k: str(v) for k, v in output.items() if k in STEM_NAMES and v}
    if isinstance(output, (list, tuple)):
        found = {}
        for item in output:
            text = str(item)
            for name in STEM_NAMES:
                if name in text.rsplit("/", 1)[-1]:
                    found[name] = text
        return found
    return {}
