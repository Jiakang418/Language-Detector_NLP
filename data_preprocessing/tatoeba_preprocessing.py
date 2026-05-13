"""
=============================================================================
D2 — Tatoeba Multilingual Sentences Pipeline (Kaggle Source)
=============================================================================
Source      : https://www.kaggle.com/datasets/alvations/tatoeba
File        : sentences_detailed.csv
              (download and place at data/raw/sentences_detailed.csv)

Raw CSV format (NO HEADER, tab-separated, 6 columns):
  [0] sentence_id   — integer, e.g. 1
  [1] lang          — ISO 639-3 code, e.g. "cmn"
  [2] text          — the sentence
  [3] username      — contributor username
  [4] \\N or value   — always null in base file, ignored
  [5] timestamp     — e.g. "2010-03-14 19:46:23"

  We use only columns [1] and [2].

Known Encoding Issue — MOJIBAKE:
  The Kaggle file is UTF-8. If pandas reads it with the wrong encoding
  (e.g. latin-1), Chinese/Japanese/Arabic characters render as garbage:
      Correct  : 我们试试看！
      Mojibake : æˆ'å€'è©¦è©¦çœ‹ï¼
  Fix: always open with encoding='utf-8'. If mojibake is already present
  in the file (i.e. the file itself was saved incorrectly), we apply
  ftfy.fix_text() to recover the original characters.

ISO Scheme  : ISO 639-1 (2-char). Tatoeba uses ISO 639-3 internally.
Register    : informal (user-contributed conversational sentences)
Script      : auto-detected via Unicode block inspection
Output      : data/processed/d2_tatoeba_processed.csv
=============================================================================
"""

import os
import re
import sys
import unicodedata

# Reconfigure stdout to UTF-8 so CJK/Arabic text prints on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

# ftfy (fixes text for you) — handles mojibake recovery
# Install: pip install ftfy
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False
    print("[D2] WARNING: ftfy not installed. Mojibake auto-fix disabled.")
    print("[D2]          Install with: pip install ftfy")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

RAW_FILE    = os.path.join(_PROJECT_ROOT, "raw_dataset", "Tatoeba", "sentences_detailed.csv")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d2_tatoeba_processed.csv")
RANDOM_SEED = 42

# Kaggle CSV column indices (0-based, tab-separated, no header)
COL_SENTENCE_ID = 0
COL_LANG        = 1
COL_TEXT        = 2
# Columns 3, 4, 5 are username / null / timestamp — not needed

TARGET_LANGUAGES_ISO1 = {
    "en": "English",
    "ms": "Malay",
    "id": "Indonesian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

# Tatoeba ISO 639-3 -> ISO 639-1 mapping
# Only target languages need entries here
ISO3_TO_ISO1 = {
    "eng": "en",
    "msa": "ms",   # Malay macrolanguage
    "zsm": "ms",   # Standard Malay (second Tatoeba code for Malay)
    "ind": "id",
    "cmn": "zh",   # Mandarin Chinese
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
}

# Target ISO 639-3 codes (used for early filtering during CSV load)
TARGET_ISO3 = set(ISO3_TO_ISO1.keys())

MAX_SAMPLES_PER_LANG = 8_000
MIN_TOKENS           = 2


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    """
    Attempt to fix mojibake encoding errors using ftfy.
    ftfy.fix_text() detects and corrects common encoding confusions,
    including the UTF-8-read-as-latin-1 pattern seen in this dataset.

    If ftfy is not available, returns text unchanged with a warning.

    Example:
        Input  : "æˆ'å€'è©¦è©¦çœ‹ï¼"   <- UTF-8 bytes read as latin-1
        Output : "我們試試看！"           <- correct Traditional Chinese
    """
    if FTFY_AVAILABLE:
        return ftfy.fix_text(text)
    return text


def detect_script(text: str) -> str:
    """
    Detect dominant script via Unicode block inspection.
    Returns one of: 'latin', 'cjk', 'arabic', 'cyrillic', 'other'.
    """
    cjk = arabic = cyrillic = latin = 0
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs + Hiragana/Katakana + Hangul
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
    """
    Whitespace split for Latin/Arabic/Cyrillic.
    Character count for CJK (no natural whitespace boundary).
    """
    if script == "cjk":
        return len([ch for ch in text if not ch.isspace()])
    return len(text.split())


def clean_text(text: str) -> str:
    """
    Tatoeba-specific cleaning.
    Order of operations:
      1. Mojibake fix (ftfy)
      2. Remove bracket/parenthetical annotations
      3. Remove leading dialogue dashes
      4. Remove ellipsis patterns
      5. Lowercase
      6. Remove digits
      7. Remove emojis and control characters
      8. Collapse whitespace
    """
    # 1. Fix mojibake before any other operation
    text = fix_encoding(text)

    # 2. Remove bracket annotations [NOTE: ...] and parentheticals
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)

    # 3. Remove leading dialogue dash
    text = re.sub(r"^\s*-\s+", "", text)

    # 4. Remove ellipses
    text = re.sub(r"\.{2,}", " ", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Remove digits
    text = re.sub(r"\d+", "", text)

    # 7. Remove emojis and control/format characters
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm", "Sk", "Cc", "Cf")
    )

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# ENCODING DIAGNOSTICS
# ---------------------------------------------------------------------------

def diagnose_encoding(raw_path: str, n_rows: int = 5) -> None:
    """
    Read the first n_rows of the raw file with both utf-8 and latin-1
    and print the results side by side. Run this first to confirm
    which encoding your specific Kaggle download uses.

    Expected output:
      utf-8  row 1 text: 我们试试看！      <- correct
      latin1 row 1 text: æˆ'ä»¬è¯•è¯•çœ‹ï¼  <- mojibake
    """
    print("[D2] === Encoding Diagnosis ===")
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        print(f"\n  Encoding: {enc}")
        try:
            with open(raw_path, encoding=enc, errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= n_rows:
                        break
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        print(f"    row {i+1} | lang={parts[1]} | text={parts[2][:60]}")
        except Exception as e:
            print(f"    Failed: {e}")
    print("\n[D2] === End Diagnosis ===\n")


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_tatoeba_kaggle(raw_path: str) -> pd.DataFrame:
    """
    Load sentences_detailed.csv from Kaggle Tatoeba dataset.

    The file is:
      - Tab-separated (\t)
      - No header row
      - UTF-8 encoded
      - ~10M+ rows — we filter on the fly to avoid loading everything

    Strategy:
      Read in chunks of 500,000 rows.
      In each chunk, immediately filter to target ISO 639-3 codes.
      This keeps memory usage manageable for the full multi-million row file.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"[D2] Raw file not found: {raw_path}\n"
            f"     Download sentences_detailed.csv from:\n"
            f"     https://www.kaggle.com/datasets/alvations/tatoeba\n"
            f"     and place it at: {raw_path}"
        )

    print(f"[D2] Loading Tatoeba from Kaggle CSV: {raw_path}")
    print(f"[D2] Filtering to target languages: {sorted(TARGET_ISO3)}")

    # Run encoding diagnosis on first load to help debug issues
    diagnose_encoding(raw_path, n_rows=3)

    CHUNK_SIZE = 500_000
    chunks_kept = []
    total_rows_seen = 0

    reader = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,                          # No header in this file
        names=["sentence_id", "lang", "text", "username", "null_col", "timestamp"],
        usecols=["lang", "text"],             # Load only the 2 columns we need
        encoding="utf-8",                     # ALWAYS utf-8 for this file
        on_bad_lines="skip",                  # Skip any malformed rows
        chunksize=CHUNK_SIZE,
        dtype={"lang": str, "text": str},
        low_memory=False,
    )

    for chunk in reader:
        total_rows_seen += len(chunk)

        # Filter to target languages immediately — reduces memory
        chunk_filtered = chunk[chunk["lang"].isin(TARGET_ISO3)].copy()

        if len(chunk_filtered) > 0:
            chunks_kept.append(chunk_filtered)

        # Progress indicator
        print(f"[D2]   Scanned {total_rows_seen:,} rows | kept {sum(len(c) for c in chunks_kept):,}", end="\r")

    print()  # newline after progress

    if not chunks_kept:
        raise ValueError(
            "[D2] No rows matched target languages after filtering.\n"
            "     Check that the 'lang' column contains ISO 639-3 codes (e.g. 'cmn', 'ind')."
        )

    df = pd.concat(chunks_kept, ignore_index=True)
    print(f"[D2] Raw rows loaded (target languages only): {len(df):,}")
    print(f"[D2] Language counts (ISO 639-3):\n{df['lang'].value_counts().to_string()}\n")

    return df


# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------

def transform_tatoeba(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline.
    Steps:
      1.  Remap ISO 639-3 -> ISO 639-1
      2.  Pre-clean deduplication (exact raw text)
      3.  Clean text (mojibake fix + noise removal)
      4.  Detect script
      5.  Count tokens
      6.  Drop rows with token_count < MIN_TOKENS
      7.  Post-clean deduplication (different raws -> same cleaned text)
      8.  Add metadata columns
      9.  Cap at MAX_SAMPLES_PER_LANG
    """

    # ---- Step 1: Remap ISO codes ----
    print("[D2] Remapping ISO 639-3 -> ISO 639-1...")
    df = df.copy()
    df["iso_code"] = df["lang"].map(ISO3_TO_ISO1)

    # Drop rows where lang mapped to nothing (shouldn't happen after filter, but safe)
    before = len(df)
    df = df[df["iso_code"].notna()].copy()
    if len(df) < before:
        print(f"[D2] Dropped {before - len(df)} rows with unmapped ISO codes")

    df.drop(columns=["lang"], inplace=True)

    # ---- Step 2: Pre-clean dedup ----
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    print(f"[D2] Pre-clean duplicates removed: {before - len(df):,}")

    # ---- Step 3: Clean text ----
    print("[D2] Cleaning text (with mojibake fix)...")
    df["text"] = df["text"].astype(str).apply(clean_text)

    # ---- Step 4: Detect script ----
    print("[D2] Detecting scripts...")
    df["script"] = df["text"].apply(detect_script)

    # ---- Step 5: Token count ----
    df["token_count"] = df.apply(
        lambda row: count_tokens(row["text"], row["script"]), axis=1
    )

    # ---- Step 6: Drop short rows ----
    before = len(df)
    df = df[df["token_count"] >= MIN_TOKENS]
    print(f"[D2] Rows dropped (token_count < {MIN_TOKENS}): {before - len(df):,}")

    # ---- Step 7: Post-clean dedup ----
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    print(f"[D2] Post-clean duplicates removed: {before - len(df):,}")

    # ---- Step 8: Metadata ----
    df["language_name"] = df["iso_code"].map(TARGET_LANGUAGES_ISO1)
    df["source"]        = "tatoeba"
    df["register"]      = "informal"

    # ---- Step 9: Cap per language ----
    df = (
        df.groupby("iso_code", group_keys=False)
          .apply(lambda g: g.sample(
              n=min(len(g), MAX_SAMPLES_PER_LANG),
              random_state=RANDOM_SEED
          ))
    )
    print(f"[D2] Rows after per-language cap: {len(df):,}")

    return df[[
        "text", "iso_code", "language_name",
        "source", "register", "token_count", "script"
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPOT-CHECK: ENCODING VERIFICATION
# ---------------------------------------------------------------------------

def spot_check_encoding(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print sample rows per language to manually verify:
      1. No mojibake remains in the text
      2. CJK scripts are correctly rendered
      3. Arabic/RTL scripts are intact

    Look for:
      GOOD: "我们试试看"  or  "saya pergi ke sekolah"
      BAD:  "æˆ'ä»¬è¯•"  or  garbled Arabic
    """
    print("\n[D2] === Spot Check: Sample rows per language ===")
    for iso, name in TARGET_LANGUAGES_ISO1.items():
        subset = df[df["iso_code"] == iso]
        if len(subset) == 0:
            print(f"\n  {name} ({iso}): NO DATA")
            continue
        samples = subset.sample(min(n, len(subset)), random_state=1)
        print(f"\n  {name} ({iso}) - {len(subset):,} rows total:")
        for _, row in samples.iterrows():
            text_preview = row['text'][:80]
            print(f"    [{row['script']}, {row['token_count']} tokens] {text_preview}")
    print("\n[D2] === End Spot Check ===\n")


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

def validate_tatoeba(df: pd.DataFrame) -> None:
    print("[D2] Validating output...")

    # No ISO 639-3 codes
    bad_iso = df[df["iso_code"].str.len() != 2]
    assert len(bad_iso) == 0, \
        f"Non ISO-639-1 codes found: {bad_iso['iso_code'].unique()}"

    # No nulls in critical columns
    for col in ["text", "iso_code", "language_name", "source", "register", "script"]:
        nulls = df[col].isna().sum()
        assert nulls == 0, f"Null values in column '{col}': {nulls}"

    # token_count >= MIN_TOKENS
    assert df["token_count"].ge(MIN_TOKENS).all(), \
        f"token_count below minimum ({MIN_TOKENS}) found"

    # Register must be 'informal' for all Tatoeba rows
    assert (df["register"] == "informal").all(), \
        "Unexpected register value in Tatoeba data"

    # No duplicate texts
    dups = df.duplicated(subset="text").sum()
    assert dups == 0, f"Duplicate texts found: {dups}"

    # Mojibake check: scan for the telltale latin-1 corruption pattern
    # ae (U+00E6) followed by specific chars is a strong mojibake signal
    mojibake_pattern = re.compile(r"[\xc3-\xef][\x80-\xbf]{2}")
    mojibake_count = df["text"].str.contains(mojibake_pattern, regex=True).sum()
    if mojibake_count > 0:
        print(f"[D2] WARNING: {mojibake_count} rows may still contain mojibake.")
        print(f"[D2]          Install ftfy to auto-fix: pip install ftfy")
        sample = df[df["text"].str.contains(mojibake_pattern, regex=True)].head(3)
        print(f"[D2]          Sample affected rows:\n{sample[['iso_code','text']].to_string()}")
    else:
        print("[D2] Mojibake check: CLEAN (OK)")

    print("[D2] Validation PASSED (OK)")
    print(f"[D2] Language distribution:\n{df['iso_code'].value_counts().to_string()}")
    print(f"[D2] Token count stats:\n{df['token_count'].describe().to_string()}\n")


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def save_tatoeba(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[D2] Saved -> {OUTPUT_FILE} ({len(df):,} rows)\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run(raw_path: str = RAW_FILE) -> pd.DataFrame:
    df_raw       = load_tatoeba_kaggle(raw_path)
    df_processed = transform_tatoeba(df_raw)
    spot_check_encoding(df_processed)   # Human-readable sanity check
    validate_tatoeba(df_processed)
    save_tatoeba(df_processed)
    return df_processed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default=RAW_FILE,
        help=f"Path to sentences_detailed.csv (default: {RAW_FILE})"
    )
    args = parser.parse_args()
    run(raw_path=args.raw)