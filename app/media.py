"""Thin ffmpeg/ffprobe wrappers. All media I/O goes through here."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"{tool} not found on PATH. Install ffmpeg (apt install ffmpeg).")
    return path


def run(args: list[str]) -> str:
    """Run a command, raising with stderr attached on failure."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{args[0]} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def probe(path: Path) -> dict:
    out = run([
        _require("ffprobe"), "-v", "error",
        "-print_format", "json", "-show_format", "-show_streams", str(path),
    ])
    return json.loads(out)


def duration(path: Path) -> float:
    info = probe(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError):
        for stream in info.get("streams", []):
            if "duration" in stream:
                return float(stream["duration"])
    raise MediaError(f"Could not determine duration of {path.name}")


def has_video_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "video" for s in probe(path).get("streams", []))


def has_audio_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Decode the first audio stream to WAV — the format every model here wants."""
    args = [
        _require("ffmpeg"), "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate),
        "-ac", "1" if mono else "2", str(dst),
    ]
    run(args)
    return dst


def slice_audio(src: Path, dst: Path, start: float, dur: float, sample_rate: int = 24000) -> Path:
    """Cut [start, start+dur) out of an audio file. Used for voice-clone references."""
    run([
        _require("ffmpeg"), "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
        "-i", str(src), "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(dst),
    ])
    return dst


def mux(
    video: Path,
    audio: Path,
    dst: Path,
    subtitles: Path | None = None,
    burn_in: bool = False,
    sub_style: str | None = None,
) -> Path:
    """Combine a video stream with a new audio track, optionally adding subtitles.

    Soft subs are muxed as a selectable track (mov_text); burn-in re-encodes the
    video with the subtitles rendered into the picture.
    """
    ffmpeg = _require("ffmpeg")

    def build(video_codec: list[str]) -> list[str]:
        args = [ffmpeg, "-y", "-i", str(video), "-i", str(audio)]
        if subtitles and burn_in:
            escaped = (
                str(subtitles).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
            )
            vf = f"subtitles='{escaped}'"
            if sub_style:
                vf += f":force_style='{sub_style}'"
            args += ["-map", "0:v:0", "-map", "1:a:0", "-vf", vf]
        elif subtitles:
            args += ["-i", str(subtitles), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
                     "-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
        else:
            args += ["-map", "0:v:0", "-map", "1:a:0"]
        return args + video_codec + ["-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]

    reencode = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    if subtitles and burn_in:
        run(build(reencode))  # burn-in always re-encodes
        return dst

    try:
        run(build(["-c:v", "copy"]))
    except MediaError:
        # Stream copy fails when the source codec can't live in MP4 (VP9, AV1,
        # some MPEG-4 variants). Re-encoding is slower but always works.
        run(build(reencode))
    return dst


def audio_only_output(audio: Path, dst: Path) -> Path:
    """For audio-only inputs: just transcode the dub to m4a."""
    run([_require("ffmpeg"), "-y", "-i", str(audio), "-c:a", "aac", "-b:a", "192k", str(dst)])
    return dst


def check_tools() -> None:
    _require("ffmpeg")
    _require("ffprobe")
