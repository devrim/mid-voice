"""SRT / WebVTT writing."""

from __future__ import annotations

import textwrap
from pathlib import Path

from .segments import Segment

MAX_LINE = 42  # characters — the usual broadcast guideline
MAX_LINES = 2


def _stamp(seconds: float, sep: str = ",") -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _wrap(text: str) -> str:
    """Wrap to at most MAX_LINE characters per line. Width is a hard cap."""
    text = " ".join(text.split())
    if len(text) <= MAX_LINE:
        return text
    return "\n".join(textwrap.wrap(text, width=MAX_LINE, break_long_words=True))


def _pick(seg: Segment, use_source: bool) -> str:
    return (seg.source_text if use_source else seg.text) or ""


def _split_cues(seg: Segment, text: str) -> list[tuple[float, float, str]]:
    """Break a segment too long to display into several cues.

    A long line has to become more cues, not wider or taller ones. The
    segment's duration is divided in proportion to each cue's character
    count, which tracks speech rate closely enough to stay in sync.
    """
    budget = MAX_LINE * MAX_LINES
    if len(text) <= budget:
        return [(seg.start, max(seg.end, seg.start + 0.3), _wrap(text))]

    words, chunks, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > budget:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)

    span = max(seg.duration, 0.3 * len(chunks))
    total = sum(len(c) for c in chunks) or 1
    cues, cursor = [], seg.start
    for chunk in chunks:
        length = span * len(chunk) / total
        cues.append((cursor, cursor + length, _wrap(chunk)))
        cursor += length
    return cues


def _cues(segments: list[Segment], use_source: bool) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        text = _pick(seg, use_source).strip()
        if text:
            out.extend(_split_cues(seg, text))
    return out


def write_srt(segments: list[Segment], dst: Path, *, use_source: bool = False) -> Path:
    blocks = [
        f"{i}\n{_stamp(start)} --> {_stamp(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(_cues(segments, use_source), start=1)
    ]
    dst.write_text("\n".join(blocks), encoding="utf-8")
    return dst


def write_vtt(segments: list[Segment], dst: Path, *, use_source: bool = False) -> Path:
    blocks = ["WEBVTT\n"] + [
        f"{_stamp(start, '.')} --> {_stamp(end, '.')}\n{text}\n"
        for start, end, text in _cues(segments, use_source)
    ]
    dst.write_text("\n".join(blocks), encoding="utf-8")
    return dst
