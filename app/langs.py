"""Whisper (ISO-639-1) -> NLLB FLORES-200 code mapping, plus display names."""

from __future__ import annotations

# Only the languages Whisper can detect that NLLB-200 also covers.
NLLB_CODES: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
    "az": "azj_Latn", "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn",
    "fr": "fra_Latn", "ga": "gle_Latn", "gl": "glg_Latn", "gu": "guj_Gujr",
    "ha": "hau_Latn", "haw": "eng_Latn", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "jw": "jav_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kn": "kan_Knda", "ko": "kor_Hang", "la": "ita_Latn", "lb": "ltz_Latn",
    "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym",
    "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn", "mt": "mlt_Latn",
    "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn", "nn": "nno_Latn",
    "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl",
    "sa": "san_Deva", "sd": "snd_Arab", "si": "sin_Sinh", "sk": "slk_Latn",
    "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn", "sq": "als_Latn",
    "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "tk": "tuk_Latn", "tl": "tgl_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn",
    "yi": "ydd_Hebr", "yo": "yor_Latn", "zh": "zho_Hans",
}

NAMES: dict[str, str] = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fa": "Persian", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

# Languages XTTS-v2 can speak. Anything else falls back to English phonetics.
XTTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh", "hu", "ko", "ja", "hi",
}


def to_nllb(code: str) -> str | None:
    return NLLB_CODES.get((code or "").lower().split("-")[0])


def name(code: str) -> str:
    code = (code or "").lower()
    return NAMES.get(code, code.upper() if code else "unknown")
