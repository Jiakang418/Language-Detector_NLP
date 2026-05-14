"""
=============================================================================
D6 — Universal Dependencies CJK Cleaning Pipeline
=============================================================================
Source      : data/processed/ud_cjk_samples.csv
              (raw extract produced by src/ud_prepare.py)
Languages   : Chinese (zh), Japanese (ja), Korean (ko)
Output      : preprocessed_dataset/d6_ud_processed.csv
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

INPUT_FILE  = os.path.join(_PROJECT_ROOT, "raw_dataset", "UD", "ud_cjk_samples.csv")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d6_ud_processed.csv")


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    print(f"[D6] Loading raw UD CJK extract: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"[D6] Raw rows loaded: {original_count:,}")

    # ---- Detect script ----
    print("[D6] Detecting scripts...")
    df["detected_script"] = df["text"].apply(detect_script)

    # ---- Clean text ----
    print("[D6] Cleaning text...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # ---- Drop empty cleaned rows ----
    df = df[df["cleaned_text"].str.strip() != ""]
    print(f"[D6] Rows dropped (empty after cleaning): {original_count - len(df):,}")

    # ---- Save ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"[D6] Saved -> {OUTPUT_FILE} ({len(df):,} rows)")
    print(f"[D6] Rows per language:\n{df['iso_code'].value_counts().to_string()}")
    print(f"[D6] Detected script counts:\n{df['detected_script'].value_counts().to_string()}")
    print(df[["text", "cleaned_text", "iso_code", "detected_script"]].head(10))
    return df


if __name__ == "__main__":
    run()
