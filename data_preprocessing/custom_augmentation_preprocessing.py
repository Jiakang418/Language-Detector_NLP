"""
=============================================================================
D7 — Custom Augmentation Dataset Cleaning Pipeline
=============================================================================
Source      : data/processed/custom_augmentation_dataset.csv
              (manually curated augmentation sentences)
Output      : preprocessed_dataset/d7_custom_augmentation_processed.csv
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

INPUT_FILE  = os.path.join(_PROJECT_ROOT, "raw_dataset", "CustomAugmentation", "custom_augmentation_dataset.csv")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "d7_custom_augmentation_processed.csv")


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    print(f"[D7] Loading raw custom augmentation data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    original_count = len(df)
    print(f"[D7] Raw rows loaded: {original_count:,}")

    # ---- Detect script ----
    print("[D7] Detecting scripts...")
    df["detected_script"] = df["text"].apply(detect_script)

    # ---- Clean text ----
    print("[D7] Cleaning text...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # ---- Save (no row-drop for augmentation — keep all rows) ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    cleaned_count = len(df[df["cleaned_text"].astype(str).str.strip() != ""])
    print(f"[D7] Saved -> {OUTPUT_FILE} ({len(df):,} rows, {cleaned_count:,} non-empty)")
    print(f"[D7] Low confidence flag distribution:\n{df['low_confidence_flag'].value_counts().to_string()}")
    print(f"[D7] Detected script counts:\n{df['detected_script'].value_counts().to_string()}")
    print(df[["text", "cleaned_text", "iso_code", "detected_script", "low_confidence_flag"]])
    return df


if __name__ == "__main__":
    run()
