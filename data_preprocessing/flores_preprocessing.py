"""
=============================================================================
D5 — FLORES-200 Validation Cleaning Pipeline
=============================================================================
Source      : data/processed/flores_validation.csv
              (raw extract produced by src/flores_prepare.py)
Languages   : English, Chinese, Japanese, Indonesian, Malay
Output      : preprocessed_dataset/d5_flores_processed.csv
=============================================================================
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — resolve paths relative to this script, not the working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Allow importing from src/
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from preprocess import preprocess_text, detect_script

INPUT_FILE  = os.path.join(_PROJECT_ROOT, "raw_dataset", "Flores", "flores_validation.csv")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d5_flores_processed.csv")


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    print(f"[D5] Loading raw FLORES validation extract: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"[D5] Raw rows loaded: {original_count:,}")

    # ---- Detect script ----
    print("[D5] Detecting scripts...")
    df["detected_script"] = df["text"].apply(detect_script)

    # ---- Clean text ----
    print("[D5] Cleaning text...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # ---- Drop empty cleaned rows ----
    df = df[df["cleaned_text"].str.strip() != ""]
    print(f"[D5] Rows dropped (empty after cleaning): {original_count - len(df):,}")

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[D5] Saved -> {OUTPUT_FILE} ({len(df):,} rows)")
    print(f"[D5] Rows per language:\n{df['iso_code'].value_counts().to_string()}")
    print(f"[D5] Detected script counts:\n{df['detected_script'].value_counts().to_string()}")
    print(df[["text", "cleaned_text", "iso_code", "detected_script"]].head(10))
    return df


if __name__ == "__main__":
    run()
