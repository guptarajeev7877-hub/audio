"""
Reissue — backend API.

Two things happen here:
  1. restore  — an ffmpeg chain that removes tape hiss, mains hum, clicks and
                clipping, then rebalances and normalises loudness.
  2. separate — Demucs splits the mix into vocals / drums / bass / other so the
                browser can rebalance the instruments in real time.

Separation runs either on Replicate (no GPU needed, costs per run) or locally
(free, needs a GPU to be quick). Set SEPARATION_BACKEND to pick.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import audio

load_dotenv()

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/reissue")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "60"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
ACCEPTED_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".wma"}

app = FastAPI(title="Reissue", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

Stage = Literal["queued", "restoring", "separating", "done", "failed"]


@dataclass
class Job:
    id: str
    mode: str
    filename: str
    stage: Stage = "queued"
    detail: str = "Waiting to start"
    error: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    highest_frequency: int | None = None
    tracks: dict[str, str] = field(default_factory=dict)  # name -> filename on disk

    @property
    def dir(self) -> Path:
        return WORK_DIR / self.id


JOBS: dict[str, Job] = {}


def _job_or_404(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "That job has expired or never existed. Upload the track again.")
    return job


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "separation_backend": audio.separation_backend(),
        "ffmpeg": audio.ffmpeg_available(),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/jobs")
async def create_job(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("separate"),
    hum_hz: int = Form(50),
    denoise: float = Form(0.5),
    presence: float = Form(0.5),
):
    if mode not in {"restore", "separate"}:
        raise HTTPException(400, "mode must be 'restore' or 'separate'")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(
            400,
            f"{suffix or 'That file type'} isn't supported. Use mp3, wav, flac, m4a, ogg or aiff.",
        )

    job = Job(id=uuid.uuid4().hex[:12], mode=mode, filename=file.filename or "track")
    job.dir.mkdir(parents=True, exist_ok=True)
    source = job.dir / f"source{suffix}"

    written = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with source.open("wb") as out:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                shutil.rmtree(job.dir, ignore_errors=True)
                raise HTTPException(413, f"That file is over the {MAX_UPLOAD_MB} MB limit.")
            out.write(chunk)

    if written == 0:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(400, "That file was empty.")

    JOBS[job.id] = job
    background.add_task(
        run_job,
        job,
        source,
        hum_hz=hum_hz,
        denoise=denoise,
        presence=presence,
    )
    return {"job_id": job.id, "mode": mode}


async def run_job(job: Job, source: Path, *, hum_hz: int, denoise: float, presence: float):
    try:
        probe = await audio.probe(source)
        job.duration = probe.duration
        job.sample_rate = probe.sample_rate
        job.highest_frequency = probe.highest_frequency

        job.stage = "restoring"
        job.detail = "Removing hiss, hum and clicks"
        cleaned = job.dir / "cleaned.wav"
        await audio.restore(
            source,
            cleaned,
            hum_hz=hum_hz,
            denoise=denoise,
            presence=presence,
        )

        if job.mode == "restore":
            final = job.dir / "restored.wav"
            await audio.normalise(cleaned, final)
            job.tracks = {"restored": final.name}
        else:
            job.stage = "separating"
            job.detail = "Splitting the mix into instruments — this is the slow part"
            stems = await audio.separate(cleaned, job.dir / "stems")
            job.tracks = {name: str(path.relative_to(job.dir)) for name, path in stems.items()}

        job.stage = "done"
        job.detail = "Ready"
    except audio.AudioError as exc:
        job.stage = "failed"
        job.error = str(exc)
        job.detail = "Failed"
    except Exception as exc:  # noqa: BLE001
        job.stage = "failed"
        job.error = f"Unexpected error: {exc}"
        job.detail = "Failed"


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _job_or_404(job_id)
    return {
        "id": job.id,
        "mode": job.mode,
        "stage": job.stage,
        "detail": job.detail,
        "error": job.error,
        "filename": job.filename,
        "duration": job.duration,
        "sample_rate": job.sample_rate,
        "highest_frequency": job.highest_frequency,
        "tracks": sorted(job.tracks.keys()),
    }


@app.get("/api/jobs/{job_id}/track/{name}")
async def get_track(job_id: str, name: str):
    job = _job_or_404(job_id)
    rel = job.tracks.get(name)
    if rel is None:
        raise HTTPException(404, f"No track called {name} in this job.")
    path = (job.dir / rel).resolve()
    if not path.is_file() or job.dir.resolve() not in path.parents:
        raise HTTPException(404, "That track is no longer on disk.")
    return FileResponse(path, media_type="audio/wav", filename=f"{name}.wav")


@app.post("/api/jobs/{job_id}/mix")
async def mix_job(job_id: str, payload: dict):
    """Render the current fader balance on the server.

    Mobile Safari can't hold four decoded stems plus an offline render buffer in
    one tab, so phones send their gains here instead of rendering locally.
    """
    job = _job_or_404(job_id)
    if job.stage != "done":
        raise HTTPException(409, "That job hasn't finished processing yet.")

    gains = payload.get("gains") or {}
    fmt = payload.get("format", "mp3")
    if fmt not in {"mp3", "wav"}:
        raise HTTPException(400, "format must be 'mp3' or 'wav'")

    inputs: list[tuple[Path, float]] = []
    for name, rel in job.tracks.items():
        path = (job.dir / rel).resolve()
        if path.is_file():
            inputs.append((path, float(gains.get(name, 1.0))))
    if not inputs:
        raise HTTPException(404, "This job has no tracks left on disk.")

    out = job.dir / f"mix.{fmt}"
    await audio.mixdown(inputs, out)
    return FileResponse(
        out,
        media_type="audio/mpeg" if fmt == "mp3" else "audio/wav",
        filename=f"reissue-mix.{fmt}",
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job = _job_or_404(job_id)
    shutil.rmtree(job.dir, ignore_errors=True)
    JOBS.pop(job_id, None)
    return {"deleted": job_id}


@app.on_event("startup")
async def sweep_old_jobs():
    """Delete anything left on disk older than JOB_TTL_MINUTES."""
    ttl = int(os.getenv("JOB_TTL_MINUTES", "90")) * 60

    async def loop():
        import time

        while True:
            now = time.time()
            for path in WORK_DIR.iterdir():
                try:
                    if path.is_dir() and now - path.stat().st_mtime > ttl:
                        shutil.rmtree(path, ignore_errors=True)
                        JOBS.pop(path.name, None)
                except OSError:
                    pass
            await asyncio.sleep(300)

    asyncio.create_task(loop())
