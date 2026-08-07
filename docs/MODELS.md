# Models and their licences

mid-voice ships no weights. It downloads them on first use into `~/.cache`
(`huggingface` and `tts` subdirectories). The code in this repository is MIT;
the weights are not, and two of the defaults are non-commercial.

| Model | Used for | Size | Licence | Commercial use |
|---|---|---|---|---|
| [Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | speech recognition, and the `whisper` translator | ~3 GB | MIT | yes |
| [NLLB-200 distilled 1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | the `nllb` translator (default) | ~5.5 GB | CC-BY-NC 4.0 | **no** |
| [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | voice cloning | ~1.8 GB | [CPML](https://coqui.ai/cpml) | **no** |

## Running a fully permissive stack

Two option changes drop both non-commercial models:

- **Translator → `Whisper translate`** — uses Whisper's own X→English task
  instead of NLLB. One model instead of two, and faster. Translation quality is
  lower, and on low-resource languages it sometimes transcribes rather than
  translates.
- **Voice → `No dub — subtitles only`** — skips XTTS entirely. You get
  translated subtitles over the original audio.

That combination is MIT end to end. You lose the dub, which is most of the
point of the tool, so treat it as the compliance path rather than the good path.

Set them as defaults without touching the UI:

```bash
MIDVOICE_WHISPER_MODEL=large-v3 ./run.sh
# then pick "Whisper translate" + "Subtitles only" in Options
```

## Swapping models

`MIDVOICE_WHISPER_MODEL` takes any
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) model name
(`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `distil-large-v3`)
or a local CTranslate2 directory.

`MIDVOICE_NLLB_MODEL` takes any NLLB checkpoint —
`facebook/nllb-200-distilled-600M` is half the size and noticeably worse;
`facebook/nllb-200-3.3B` is better and needs ~8 GB of VRAM.

`MIDVOICE_XTTS_MODEL` takes any Coqui TTS model id, but the pipeline assumes
XTTS-style zero-shot cloning (`speaker_wav` conditioning). A non-cloning model
will not work without changing `app/tts.py`.

## Where the downloads go

```
~/.cache/huggingface/hub/          Whisper, NLLB
~/.local/share/tts/                XTTS-v2
```

Delete those directories to reclaim ~10 GB; the next run re-downloads whatever
it needs.
