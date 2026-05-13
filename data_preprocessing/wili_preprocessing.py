"""
=============================================================================
D1 — WiLI-2018 Dataset Pipeline
=============================================================================
Source      : https://huggingface.co/datasets/wili_2018
              OR direct download: https://zenodo.org/record/841984
ISO Scheme  : ISO 639-1 (2-char). WiLI uses ISO 639-3 internally;
              we remap to ISO 639-1 for all target languages.
Register    : formal (Wikipedia prose)
Script      : auto-detected via Unicode block inspection
Output      : data/processed/d1_wili_processed.csv
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

RAW_DIR     = os.path.join(_PROJECT_ROOT, "raw_dataset", "Wili2018")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d1_wili_processed.csv")
RANDOM_SEED = 42

# Target languages for this project (ISO 639-1 code → full name)
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

# WiLI uses ISO 639-3. Map WiLI codes → ISO 639-1 for target languages.
# Full WiLI label set has 235 entries; we only need to remap our targets.
WILI_ISO3_TO_ISO1 = {
    "eng": "en",
    "msa": "ms",   # Malay macrolanguage
    "zsm": "ms",   # Standard Malay (alternate WiLI code)
    "ind": "id",
    "zho": "zh",
    "cmn": "zh",   # Mandarin Chinese
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
}

# Max samples per language to keep corpus balanced
MAX_SAMPLES_PER_LANG = 10_000


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """
    Detect the dominant script of a text string by inspecting Unicode blocks.
    Returns one of: 'latin', 'cjk', 'arabic', 'cyrillic', 'other'.
    """
    cjk     = 0
    arabic  = 0
    cyrillic = 0
    latin   = 0

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


def clean_text(text: str) -> str:
    """
    Normalise and clean a raw text string:
    - Lowercase
    - Remove digits
    - Remove emojis and non-printable characters
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    """
    # Lowercase
    text = text.lower()
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Remove emojis and symbols (keep letters, basic punctuation, spaces)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm", "Sk", "Cc", "Cf")
    )
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_tokens(text: str, script: str) -> int:
    """
    Count tokens: whitespace split for Latin/Arabic/Cyrillic,
    character count for CJK (no natural whitespace boundary).
    """
    if script == "cjk":
        return len([ch for ch in text if not ch.isspace()])
    return len(text.split())

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_wili() -> pd.DataFrame:
    """
    Load WiLI-2018 from local text files, stitch X (text) and Y (labels),
    and combine the train and test splits into a single pooled DataFrame.
    """
    print(f"[D1] Loading WiLI-2018 from local path: {RAW_DIR}...")
    
    with open(os.path.join(RAW_DIR, 'x_train.txt'), 'r', encoding='utf-8') as f:
        x_train = [line.strip() for line in f]
    with open(os.path.join(RAW_DIR, 'y_train.txt'), 'r', encoding='utf-8') as f:
        y_train = [line.strip() for line in f]
        
    train_df = pd.DataFrame({"text": x_train, "iso_code": y_train})
    
    # 2. Load and stitch TEST files
    with open(os.path.join(RAW_DIR, 'x_test.txt'), 'r', encoding='utf-8') as f:
        x_test = [line.strip() for line in f]
    with open(os.path.join(RAW_DIR, 'y_test.txt'), 'r', encoding='utf-8') as f:
        y_test = [line.strip() for line in f]
        
    test_df = pd.DataFrame({"text": x_test, "iso_code": y_test})
    
    # 3. Pool Train and Test together
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"[D1] Raw rows loaded (Train + Test pooled): {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------

def transform_wili(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline for WiLI data.
    Steps:
      1. Remap ISO 639-3 codes to ISO 639-1
      2. Filter to target languages only
      3. Clean text
      4. Detect script
      5. Count tokens
      6. Add metadata columns
      7. Drop rows with empty text or token_count < 3
      8. Cap at MAX_SAMPLES_PER_LANG per language
    """

    # ---- Step 1: Remap ISO codes ----
    print("[D1] Remapping ISO 639-3 -> ISO 639-1...")
    df["iso_code"] = df["iso_code"].map(WILI_ISO3_TO_ISO1)

    # ---- Step 2: Filter target languages ----
    df = df[df["iso_code"].notna()].copy()
    df = df[df["iso_code"].isin(TARGET_LANGUAGES_ISO1.keys())].copy()
    print(f"[D1] Rows after language filter: {len(df):,}")

    # ---- Step 3: Clean text ----
    print("[D1] Cleaning text...")
    df["text"] = df["text"].astype(str).apply(clean_text)

    # ---- Step 4: Detect script ----
    print("[D1] Detecting scripts...")
    df["script"] = df["text"].apply(detect_script)

    # ---- Step 5: Count tokens ----
    df["token_count"] = df.apply(
        lambda row: count_tokens(row["text"], row["script"]), axis=1
    )

    # ---- Step 6: Metadata columns ----
    df["language_name"] = df["iso_code"].map(TARGET_LANGUAGES_ISO1)
    df["source"]        = "wili"
    df["register"]      = "formal"

    # ---- Step 7: Quality filters ----
    before = len(df)
    df = df[df["text"].str.strip().ne("")]
    df = df[df["token_count"] >= 3]
    print(f"[D1] Rows dropped by quality filter: {before - len(df):,}")

    # ---- Step 8: Cap per language ----
    df = (
        df.groupby("iso_code", group_keys=False)
          .apply(lambda g: g.sample(
              n=min(len(g), MAX_SAMPLES_PER_LANG),
              random_state=RANDOM_SEED
          ))
    )
    print(f"[D1] Rows after per-language cap: {len(df):,}")

    # ---- Reorder columns to unified schema ----
    return df[[
        "text", "iso_code", "language_name",
        "source", "register", "token_count", "script"
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

def validate_wili(df: pd.DataFrame) -> None:
    """
    Assert the output conforms to unified schema requirements.
    Raises AssertionError with a descriptive message on any violation.
    """
    print("[D1] Validating output...")

    # No 3-char ISO codes (ISO 639-3 leak)
    three_char = df[df["iso_code"].str.len() != 2]
    assert len(three_char) == 0, \
        f"ISO 639-3 codes found: {three_char['iso_code'].unique()}"

    # No null values in critical columns
    for col in ["text", "iso_code", "language_name", "source", "register", "script"]:
        nulls = df[col].isna().sum()
        assert nulls == 0, f"Null values in column '{col}': {nulls}"

    # Register must be one of valid values
    valid_registers = {"formal", "informal", "mixed", "unknown"}
    bad_register = set(df["register"].unique()) - valid_registers
    assert not bad_register, f"Invalid register values: {bad_register}"

    # Script must be one of valid values
    valid_scripts = {"latin", "cjk", "arabic", "cyrillic", "other"}
    bad_script = set(df["script"].unique()) - valid_scripts
    assert not bad_script, f"Invalid script values: {bad_script}"

    # token_count must be positive integer
    assert df["token_count"].gt(0).all(), "token_count has zero or negative values"

    print("[D1] Validation PASSED (OK)")
    print(f"[D1] Language distribution:\n{df['iso_code'].value_counts().to_string()}\n")


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def save_wili(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[D1] Saved -> {OUTPUT_FILE} ({len(df):,} rows)\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    df_raw       = load_wili()
    df_processed = transform_wili(df_raw)
    validate_wili(df_processed)
    save_wili(df_processed)
    return df_processed


if __name__ == "__main__":
    run()