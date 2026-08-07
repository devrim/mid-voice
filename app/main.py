"""FastAPI app: upload a video, watch it translate, download the result."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import config, jobs, langs, media, pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("midvoice")

STATIC = Path(__file__).parent / "static"
CHUNK = 1024 * 1024
ALLOWED_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
}

app = FastAPI(title="mid-voice", docs_url="/api/docs")
queue = jobs.JobQueue(pipeline.run)


# --- capability probe ------------------------------------------------------
def _capabilities() -> dict:
    caps = {"cuda": False, "gpu": None, "tts": False, "nllb": False, "ffmpeg": True}
    try:
        media.check_tools()
    except media.MediaError:
        caps["ffmpeg"] = False
    try:
        import torch

        caps["cuda"] = torch.cuda.is_available()
        if caps["cuda"]:
            caps["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import TTS  # noqa: F401

        caps["tts"] = True
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401

        caps["nllb"] = True
    except ImportError:
        pass
    return caps


# --- routes ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "capabilities": _capabilities(),
        "whisper_model": config.WHISPER_MODEL,
        "languages": [{"code": c, "name": n} for c, n in sorted(langs.NAMES.items(), key=lambda x: x[1])],
        "max_speedup": config.MAX_SPEEDUP,
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile,
    options: str = Form("{}"),
) -> dict:
    name = Path(file.filename or "input").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}")

    try:
        opts = json.loads(options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Bad options JSON: {exc}") from exc

    job_id_dir = config.JOBS_DIR / _new_dir_name()
    job_id_dir.mkdir(parents=True, exist_ok=True)
    dest = job_id_dir / f"input{suffix}"

    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(CHUNK):
            size += len(chunk)
            fh.write(chunk)
    if size == 0:
        shutil.rmtree(job_id_dir, ignore_errors=True)
        raise HTTPException(400, "Empty upload")

    job = queue.submit(name, dest, _normalize_options(opts))
    log.info("queued %s (%s, %.1f MB)", job.id[:8], name, size / 1e6)
    return job.to_dict()


def _new_dir_name() -> str:
    import uuid

    return uuid.uuid4().hex


def _normalize_options(opts: dict) -> dict:
    def pick(key: str, allowed: set[str], default: str) -> str:
        value = str(opts.get(key, default))
        return value if value in allowed else default

    return {
        "subtitles": pick("subtitles", {"none", "soft", "burn"}, "soft"),
        "subtitle_lang": pick("subtitle_lang", {"en", "both"}, "en"),
        "voice": pick("voice", {"clone", "none"}, "clone"),
        "mix_mode": pick("mix_mode", {"duck", "overlay", "replace"}, "duck"),
        "translator": pick("translator", {"auto", "nllb", "whisper"}, "auto"),
        "source_lang": str(opts.get("source_lang") or "").strip(),
        "max_speedup": max(1.0, min(2.0, float(opts.get("max_speedup") or config.MAX_SPEEDUP))),
    }


@app.get("/api/jobs")
async def list_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in queue.all()[:50]]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")

    async def stream():
        last = -1
        while True:
            if job.version != last:
                last = job.version
                yield f"data: {json.dumps(job.to_dict())}\n\n"
            if job.status in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    job.cancel()
    return job.to_dict()


@app.get("/api/jobs/{job_id}/files/{name}")
async def get_file(job_id: str, name: str) -> FileResponse:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    # Resolve inside the job directory only — never let `name` escape it.
    base = job.source_path.parent.resolve()
    target = (base / Path(name).name).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target, filename=target.name)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    job = queue.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    job.cancel()
    shutil.rmtree(job.source_path.parent, ignore_errors=True)
    return {"deleted": job_id}


def main() -> None:
    import uvicorn

    caps = _capabilities()
    print(f"\n  mid-voice → http://{config.HOST}:{config.PORT}")
    print(f"  GPU: {caps['gpu'] or 'CPU only'} | voice cloning: "
          f"{'yes' if caps['tts'] else 'not installed'} | NLLB: "
          f"{'yes' if caps['nllb'] else 'not installed'}\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
