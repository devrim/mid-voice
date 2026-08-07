"""The unit of work that flows through the whole pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    source_text: str
    text: str = ""  # English translation; filled in by the translate step
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "source_text": self.source_text,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class Transcript:
    language: str
    language_probability: float = 0.0
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "segments": [s.to_dict() for s in self.segments],
        }
