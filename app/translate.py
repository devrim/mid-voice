"""Translation backends.

Two ways to get English out of a non-English video:

* ``whisper``  — a second Whisper pass with task="translate". No extra models,
  no extra deps, timings come straight from the decoder. Weaker on low-resource
  languages, and occasionally just transcribes instead of translating.
* ``nllb``     — Meta's NLLB-200 applied per segment. Better translation
  quality, and it reuses the transcribe pass's timings exactly, so the source
  and English subtitle tracks line up line-for-line.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable

from . import config, langs
from .segments import Segment

log = logging.getLogger(__name__)

_pipe: tuple = ()
_lock = threading.Lock()


class TranslationError(RuntimeError):
    pass


def _load_nllb():
    """Load NLLB once and cache it."""
    global _pipe
    with _lock:
        if _pipe:
            return _pipe
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise TranslationError(
                "NLLB backend needs transformers + torch. "
                'Install with: uv pip install -e ".[mt]"'
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        log.info("loading %s on %s", config.NLLB_MODEL, device)
        tok = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL, torch_dtype=dtype)
        model.to(device).eval()
        _pipe = (tok, model, device)
    return _pipe


def _bos_id(tok, code: str) -> int:
    """Get the forced-BOS token for a target language.

    transformers moved this around between versions, so try both spellings.
    """
    legacy = getattr(tok, "lang_code_to_id", None)
    if isinstance(legacy, dict) and code in legacy:
        return legacy[code]
    tid = tok.convert_tokens_to_ids(code)
    if tid is None or tid == tok.unk_token_id:
        raise TranslationError(f"NLLB tokenizer does not know language code {code!r}")
    return tid


def translate_nllb(
    segments: list[Segment],
    source_lang: str,
    *,
    batch_size: int = 8,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    """Fill in ``Segment.text`` for each segment, in place."""
    src = langs.to_nllb(source_lang)
    if src is None:
        raise TranslationError(
            f"NLLB has no code for language {source_lang!r}; use the Whisper backend instead."
        )
    if src == "eng_Latn":
        for s in segments:
            s.text = s.source_text
        return

    import torch  # noqa: PLC0415

    tok, model, device = _load_nllb()
    tok.src_lang = src
    bos = _bos_id(tok, "eng_Latn")

    done = 0
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        enc = tok([s.source_text for s in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                max_new_tokens=384,
                num_beams=4,
                no_repeat_ngram_size=4,
            )
        for seg, text in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
            seg.text = text.strip()
        done += len(batch)
        if progress:
            progress(done / len(segments), batch[-1].text)


_REPEAT = re.compile(r"\b(\w[\w']*)(?:[\s,]+\1\b){3,}", re.IGNORECASE)


def clean(text: str) -> str:
    """Trim the artefacts these models produce on silence and music."""
    text = " ".join(text.split())
    text = _REPEAT.sub(r"\1", text)  # "the the the the" -> "the"
    return text.strip()


def unload() -> None:
    global _pipe
    with _lock:
        _pipe = ()
