"""
=============================================================================
D4 — OPUS OpenSubtitles Preprocessing Pipeline
=============================================================================
Source      : raw_dataset/opus_corpora/*.txt  (monolingual subtitle dumps)
Register    : informal (movie/TV subtitles)
ISO Scheme  : ISO 639-1 (2-char)
Script      : auto-detected via Unicode block inspection
Output      : preprocessed_dataset/d4_opus_processed.csv
=============================================================================
"""

import os
import re
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d4_opus_processed.csv")

MAX_SAMPLES_PER_LANG = 8000
MIN_TOKENS           = 2
MAX_TOKENS           = 30
RANDOM_SEED          = 42

OPUS_FILE_MAP = {
    "en": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "en.txt"),      "English"),
    "fr": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "fr.txt"),      "French"),
    "de": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "de.txt"),      "German"),
    "es": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "es.txt"),      "Spanish"),
    "ar": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "ar.txt"),      "Arabic"),
    "zh": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "zh_cn.txt"),   "Chinese"),
    "ja": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "ja.txt"),      "Japanese"),
    "ko": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "ko.txt"),      "Korean"),
    "id": (os.path.join(_PROJECT_ROOT, "raw_dataset", "opus_corpora", "id.txt"),      "Indonesian"),
}

# Pre-compiled regex patterns
_RE_MUSIC       = re.compile(r"[♪♫♬♩]")
_RE_MUSIC_HASH  = re.compile(r"(?:^|\s)#\s*|#(?:\s|$)")
_RE_ALLCAPS     = re.compile(r"\b[A-Z]{3,}\b")
_RE_BRACKETS    = re.compile(r"\[.*?\]")
_RE_PARENS      = re.compile(r"\(.*?\)")
_RE_LEADING_DASH= re.compile(r"^\s*-\s+")
_RE_SPEAKER     = re.compile(r"^[A-Z][A-Za-z.\s]{0,20}:\s+")
_RE_ELLIPSIS    = re.compile(r"\.{2,}")
_RE_HTML_ENT    = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_RE_POS_TAG     = re.compile(r"\{\\an\d+\}")
_RE_HTML_TAG    = re.compile(r"<[^>]+>")
_RE_DIGITS      = re.compile(r"\d+")
_RE_MULTI_WS    = re.compile(r"\s+")
_RE_AR_INDIC    = re.compile(r"[٠١٢٣٤٥٦٧٨٩]")
_RE_AR_DIR      = re.compile(r"[\u200F\u200E\u202B\u202C]")
_RE_AR_DIAC     = re.compile(r"[\u064B-\u065F]")
_RE_ZH_FWPUNCT  = re.compile(r"[！？]")
_RE_INV_PUNCT   = re.compile(r"[¿¡]")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """Detect dominant script via Unicode block inspection."""
    cjk = arabic = cyrillic = latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            cjk += 1
        elif 0x0600 <= cp <= 0x06FF:
            arabic += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif 0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F:
            latin += 1
    scores = {"cjk": cjk, "arabic": arabic, "cyrillic": cyrillic, "latin": latin}
    dominant = max(scores, key=scores.get)
    return dominant if scores[dominant] > 0 else "other"


def count_tokens(text: str, script: str) -> int:
    """Word count for Latin/Arabic; character count for CJK."""
    if script == "cjk":
        return len([ch for ch in text if not ch.isspace()])
    return len(text.split())


# ---------------------------------------------------------------------------
# STAGE 1 — SUBTITLE NOISE REMOVAL
# ---------------------------------------------------------------------------

def remove_subtitle_noise(text: str) -> str:
    """Universal subtitle artefact removal (all languages)."""
    text = _RE_MUSIC.sub("", text)
    text = _RE_MUSIC_HASH.sub(" ", text)
    text = _RE_ALLCAPS.sub("", text)
    text = _RE_BRACKETS.sub("", text)
    text = _RE_PARENS.sub("", text)
    text = _RE_LEADING_DASH.sub("", text)
    text = _RE_SPEAKER.sub("", text)
    text = _RE_ELLIPSIS.sub(" ", text)
    text = _RE_HTML_ENT.sub("", text)
    text = _RE_POS_TAG.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    return text


# ---------------------------------------------------------------------------
# STAGE 2 — LANGUAGE-SPECIFIC CLEANING
# ---------------------------------------------------------------------------

def clean_latin(text: str) -> str:
    """Latin-script languages: en, fr, de, es, id."""
    text = text.lower()
    text = _RE_INV_PUNCT.sub("", text)
    return text


def clean_arabic(text: str) -> str:
    """Arabic-specific cleaning."""
    text = _RE_AR_INDIC.sub("", text)
    text = _RE_AR_DIR.sub("", text)
    text = text.replace("\u0640", "")      # tatweel / kashida
    text = _RE_AR_DIAC.sub("", text)
    return text


def clean_cjk(text: str, iso_code: str) -> str:
    """CJK cleaning dispatched by language."""
    if iso_code in ("zh", "ja"):
        text = _RE_ZH_FWPUNCT.sub("", text)
    if iso_code == "ja":
        text = text.replace("\u30FB", "")  # katakana middle dot
    if iso_code == "ko":
        text = re.sub(r"[\u3130-\u318F]", "", text)  # compatibility jamo
    return text


# ---------------------------------------------------------------------------
# STAGE 3 — UNIVERSAL CLEANING
# ---------------------------------------------------------------------------

def clean_universal(text: str) -> str:
    """Final pass applied to every language after language-specific cleaning."""
    text = _RE_DIGITS.sub("", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm", "Sk", "Cc", "Cf")
    )
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

def clean_text(text: str, iso_code: str) -> str:
    """Full cleaning pipeline: subtitle noise → language-specific → universal."""
    text = remove_subtitle_noise(text)
    if iso_code in ("en", "fr", "de", "es", "id"):
        text = clean_latin(text)
    elif iso_code == "ar":
        text = clean_arabic(text)
    elif iso_code in ("zh", "ja", "ko"):
        text = clean_cjk(text, iso_code)
    text = clean_universal(text)
    return text


# ---------------------------------------------------------------------------
# QUALITY FILTER
# ---------------------------------------------------------------------------

def _passes_quality(text: str, token_count: int) -> bool:
    """Return True if the row should be kept."""
    if not text:
        return False
    if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
        return False
    # More than 60 % non-alphabetic → reject
    alpha = sum(1 for ch in text if ch.isalpha())
    if len(text) > 0 and alpha / len(text) < 0.40:
        return False
    return True


# ---------------------------------------------------------------------------
# PER-LANGUAGE LOADER
# ---------------------------------------------------------------------------

def load_and_process_language(iso: str, filepath: str,
                              language_name: str) -> pd.DataFrame:
    """Stream a single OPUS file, clean, filter, deduplicate, sample."""
    print(f"\n[D4] === {language_name} ({iso}) ===")
    print(f"[D4] Reading: {filepath}")

    if not os.path.exists(filepath):
        print(f"[D4] WARNING: File not found — {filepath}. Skipping {language_name}.")
        return pd.DataFrame()

    # ---- Read lines (generator for memory efficiency) ----
    try:
        raw_lines = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    raw_lines.append(stripped)
        print(f"[D4] Non-empty lines read: {len(raw_lines):,}")
    except Exception as e:
        print(f"[D4] ERROR reading {filepath}: {e}. Skipping {language_name}.")
        return pd.DataFrame()

    if not raw_lines:
        print(f"[D4] WARNING: {language_name} file is empty after reading. Skipping.")
        return pd.DataFrame()

    # ---- Pre-clean dedup (raw exact duplicates within this language) ----
    before = len(raw_lines)
    raw_lines = list(dict.fromkeys(raw_lines))  # preserves order
    print(f"[D4] Pre-clean duplicates removed: {before - len(raw_lines):,}")

    # ---- Reservoir sampling: if file has >> MAX_SAMPLES_PER_LANG * 10 lines,
    #      pre-sample to keep memory manageable ----
    pre_sample_cap = MAX_SAMPLES_PER_LANG * 12
    if len(raw_lines) > pre_sample_cap:
        import random
        rng = random.Random(RANDOM_SEED)
        raw_lines = rng.sample(raw_lines, pre_sample_cap)
        print(f"[D4] Pre-sampled to {pre_sample_cap:,} lines (memory optimisation)")

    # ---- Clean ----
    print(f"[D4] Cleaning text...")
    cleaned = [clean_text(line, iso) for line in raw_lines]

    df = pd.DataFrame({"text": cleaned})
    after_noise = len(df)
    print(f"[D4] Rows after subtitle noise removal: {after_noise:,}")

    # ---- Post-clean dedup ----
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    print(f"[D4] Post-clean duplicates removed: {before - len(df):,}")

    # ---- Script detection & token counting ----
    df["script"]      = df["text"].apply(detect_script)
    df["token_count"] = df.apply(
        lambda r: count_tokens(r["text"], r["script"]), axis=1
    )

    # ---- Quality filter ----
    before = len(df)
    mask = df.apply(
        lambda r: _passes_quality(r["text"], r["token_count"]), axis=1
    )
    df = df[mask].copy()
    print(f"[D4] Rows dropped by quality filter: {before - len(df):,}")

    if df.empty:
        print(f"[D4] WARNING: {language_name} has 0 rows after filtering. Skipping.")
        return pd.DataFrame()

    # ---- Add metadata columns ----
    df["iso_code"]      = iso
    df["language_name"]  = language_name
    df["source"]         = "opus"
    df["register"]       = "informal"

    # ---- Sample ----
    if len(df) > MAX_SAMPLES_PER_LANG:
        df = df.sample(n=MAX_SAMPLES_PER_LANG, random_state=RANDOM_SEED)
    print(f"[D4] Final rows for {language_name}: {len(df):,}")

    # ---- Reorder to unified schema ----
    return df[[
        "text", "iso_code", "language_name",
        "source", "register", "token_count", "script"
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """Assert output conforms to the unified schema."""
    print("\n[D4] Validating output...")
    expected_langs = set(OPUS_FILE_MAP.keys())
    errors = []

    # No nulls
    for col in ["text","iso_code","language_name","source","register","token_count","script"]:
        n = df[col].isna().sum()
        if n > 0:
            errors.append(f"  Null values in '{col}': {n}")

    # ISO 639-1 (2 chars)
    bad = df[df["iso_code"].str.len() != 2]
    if len(bad):
        errors.append(f"  Non-2-char iso_code values: {bad['iso_code'].unique()}")

    # Expected language set
    present = set(df["iso_code"].unique())
    missing = expected_langs - present
    if missing:
        errors.append(f"  Missing languages: {missing}")
    unexpected = present - expected_langs
    if unexpected:
        errors.append(f"  Unexpected languages: {unexpected}")

    # Constant columns
    if not (df["register"] == "informal").all():
        errors.append("  Register column has values other than 'informal'")
    if not (df["source"] == "opus").all():
        errors.append("  Source column has values other than 'opus'")

    # Script values
    valid_scripts = {"latin", "cjk", "arabic", "cyrillic", "other"}
    bad_s = set(df["script"].unique()) - valid_scripts
    if bad_s:
        errors.append(f"  Invalid script values: {bad_s}")

    # Token bounds
    if not df["token_count"].ge(MIN_TOKENS).all():
        errors.append(f"  token_count below {MIN_TOKENS} found")
    if not df["token_count"].le(MAX_TOKENS).all():
        errors.append(f"  token_count above {MAX_TOKENS} found")

    # No within-language duplicates
    for iso in df["iso_code"].unique():
        subset = df[df["iso_code"] == iso]
        dups = subset.duplicated(subset="text").sum()
        if dups:
            errors.append(f"  {iso}: {dups} duplicate texts")

    if errors:
        print("[D4] Validation FAILED:")
        for e in errors:
            print(e)
    else:
        print("[D4] Validation PASSED (OK)")

    print(f"[D4] Language distribution:\n{df['iso_code'].value_counts().to_string()}")
    print(f"[D4] Token count stats:\n{df['token_count'].describe().to_string()}\n")


# ---------------------------------------------------------------------------
# SPOT CHECK
# ---------------------------------------------------------------------------

def spot_check(df: pd.DataFrame, n: int = 3) -> None:
    """Print sample rows per language for manual verification."""
    print("\n[D4] === Spot Check: Sample rows per language ===")
    print("GOOD if CJK shows readable characters, BAD if you see")
    print("garbled latin characters like æˆ' in Chinese rows.\n")
    for iso, name in sorted(OPUS_FILE_MAP.items(), key=lambda x: x[1]):
        subset = df[df["iso_code"] == iso]
        if subset.empty:
            print(f"  {name} ({iso}): NO DATA")
            continue
        samples = subset.sample(min(n, len(subset)), random_state=1)
        print(f"  {name} ({iso}) — {len(subset):,} rows:")
        for _, row in samples.iterrows():
            preview = row["text"][:80]
            print(f"    [{row['script']}, {row['token_count']} tok] {preview}")
        print()
    print("[D4] === End Spot Check ===\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """Process all OPUS files, concatenate, validate, save."""
    print("=" * 65)
    print("D4 — OPUS OpenSubtitles Preprocessing Pipeline")
    print("=" * 65)

    frames = []
    for iso, (filepath, lang_name) in OPUS_FILE_MAP.items():
        df_lang = load_and_process_language(iso, filepath, lang_name)
        if not df_lang.empty:
            frames.append(df_lang)

    if not frames:
        print("[D4] FATAL: No languages produced output. Aborting.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"\n[D4] Total rows across all languages: {len(df):,}")

    # Validate & spot check
    validate(df)
    spot_check(df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[D4] Saved -> {OUTPUT_FILE} ({len(df):,} rows)\n")

    return df


if __name__ == "__main__":
    run()
