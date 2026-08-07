"""In-memory job registry and a single-worker queue.

One worker on purpose: ASR and TTS both want the GPU, and running two jobs at
once on one card is slower than running them back to back.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

STEPS: list[tuple[str, str, float]] = [
    ("probe", "Inspecting file", 0.01),
    ("extract", "Extracting audio", 0.03),
    ("transcribe", "Transcribing", 0.30),
    ("translate", "Translating", 0.12),
    ("synthesize", "Synthesizing voice", 0.40),
    ("mix", "Mixing audio", 0.06),
    ("mux", "Writing output", 0.08),
]
_WEIGHTS = {key: weight for key, _, weight in STEPS}
_ORDER = [key for key, _, _ in STEPS]


class Cancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    filename: str
    source_path: Path
    options: dict[str, Any]
    status: str = "queued"  # queued | running | done | error | cancelled
    step: str = ""
    step_label: str = "Waiting in queue"
    step_progress: float = 0.0
    detail: str = ""
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    version: int = 0  # bumped on every change so SSE can skip idle pushes
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- progress reporting -------------------------------------------------
    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        done = 0.0
        for key in _ORDER:
            if key == self.step:
                return min(1.0, done + _WEIGHTS[key] * max(0.0, min(1.0, self.step_progress)))
            done += _WEIGHTS[key]
        return min(1.0, done)

    def set_step(self, key: str, detail: str = "") -> None:
        self.check_cancelled()
        label = next((lbl for k, lbl, _ in STEPS if k == key), key)
        with self._lock:
            self.step, self.step_label = key, label
            self.step_progress, self.detail = 0.0, detail
            self.version += 1
        self.log(label + (f" — {detail}" if detail else ""))

    def set_progress(self, fraction: float, detail: str = "") -> None:
        self.check_cancelled()
        with self._lock:
            self.step_progress = max(0.0, min(1.0, fraction))
            if detail:
                self.detail = detail[:200]
            self.version += 1

    def log(self, message: str) -> None:
        with self._lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            del self.logs[:-400]
            self.version += 1
        log.info("job %s: %s", self.id[:8], message)

    # -- cancellation -------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        if self.status == "queued":
            self.status = "cancelled"
            self.step_label = "Cancelled"
            self.version += 1

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("cancelled by user")

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        # The transcript can be thousands of lines; re-sending it on every SSE
        # tick during synthesis would dwarf the actual progress payload.
        result = self.result
        if self.status not in ("done", "error", "cancelled"):
            result = {k: v for k, v in result.items() if k not in ("segments", "source_segments")}
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "step": self.step,
            "step_label": self.step_label,
            "progress": round(self.progress, 4),
            "detail": self.detail,
            "error": self.error,
            "options": self.options,
            "result": result,
            "logs": self.logs[-60:],
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
            "version": self.version,
        }


class JobQueue:
    def __init__(self, runner: Callable[[Job], dict]) -> None:
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="midvoice-worker")
        self._worker.start()

    def submit(self, filename: str, source_path: Path, options: dict) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, source_path=source_path, options=options)
        self._jobs[job.id] = job
        self._q.put(job.id)
        depth = self._q.qsize()
        job.step_label = "Waiting in queue" if depth > 1 else "Starting"
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            job.status, job.started = "running", time.time()
            job.version += 1
            try:
                job.result = self._runner(job)
                job.status = "done"
                job.step_label = "Finished"
                job.step_progress = 1.0
                job.log("Done")
            except Cancelled:
                job.status = "cancelled"
                job.step_label = "Cancelled"
                job.log("Cancelled")
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                job.status = "error"
                job.error = str(exc)
                job.step_label = "Failed"
                job.log(f"ERROR: {exc}")
                log.exception("job %s failed", job.id[:8])
            finally:
                job.finished = time.time()
                job.version += 1
