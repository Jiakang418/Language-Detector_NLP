import pandas as pd
import os

import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG (Updated for Absolute Pathing)
# ---------------------------------------------------------------------------
# Automatically find the directory this script is saved in (e.g., 'data_preprocessing')
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the main project root (e.g., 'Language-Detector_NLP')
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Dynamically build the absolute path to your dataset folder
INPUT_DIR = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE = os.path.join(INPUT_DIR, "unified_corpus.csv")

# We officially expand to an 8-column schema to accommodate D7's critical flag
EXPECTED_COLUMNS = [
    "text", "iso_code", "language_name", 
    "source", "register", "token_count", "script", "low_confidence_flag"
]

# We officially expand to an 8-column schema to accommodate D7's critical flag
EXPECTED_COLUMNS = [
    "text", "iso_code", "language_name", 
    "source", "register", "token_count", "script", "low_confidence_flag"
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def count_tokens(row):
    """Recalculate tokens since D5/D6/D7 are missing this column."""
    text = str(row["text"])
    script = str(row["script"]).lower()
    
    # D5/D6 output "chinese", "latin", etc., so we handle standard CJK normalization
    if script in ["cjk", "chinese", "japanese", "korean"]:
        return len([c for c in text if not c.isspace()])
    return len(text.split())

def process_standard_df(filename):
    """Processes D1, D2, D3, D4 which already follow the schema."""
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    
    df = pd.read_csv(filepath, keep_default_na=False)
    
    # Backfill the 8th column for standard datasets
    if "low_confidence_flag" not in df.columns:
        df["low_confidence_flag"] = False
        
    return df[EXPECTED_COLUMNS].copy()

def process_anomalous_df(filename, default_register):
    """Processes D5, D6, D7 which have schema deviations."""
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] {filename} not found.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
        
    df = pd.read_csv(filepath, keep_default_na=False)
    
    # 1. Promote cleaned_text to the primary text feature
    if "cleaned_text" in df.columns:
        df["text"] = df["cleaned_text"]
        
    # 2. Rename detected_script to script and normalize naming
    if "detected_script" in df.columns:
        df["script"] = df["detected_script"]
    df["script"] = df["script"].str.lower().replace({"chinese": "cjk", "japanese": "cjk", "korean": "cjk"})
        
    # 3. Inject missing register
    if "register" not in df.columns:
        df["register"] = default_register
        
    # 4. Inject missing token count
    if "token_count" not in df.columns:
        df["token_count"] = df.apply(count_tokens, axis=1)
        
    # 5. Handle the low confidence flag (ensure D7's True is read as a boolean)
    if "low_confidence_flag" in df.columns:
        df["low_confidence_flag"] = df["low_confidence_flag"].astype(str).str.lower() == "true"
    else:
        df["low_confidence_flag"] = False
        
    return df[EXPECTED_COLUMNS].copy()

# ---------------------------------------------------------------------------
# EXECUTION PIPELINE
# ---------------------------------------------------------------------------
print("=== Stage 1: Corpus Unification & Standardization ===")

# Load Standard Datasets
df_d1 = process_standard_df("d1_wili_processed.csv")
df_d2 = process_standard_df("d2_tatoeba_processed.csv")
df_d3 = process_standard_df("d3_cc100_processed.csv")
df_d4 = process_standard_df("d4_opus_processed.csv")

# Load Anomalous Datasets (Applying correct registers)
df_d5 = process_anomalous_df("d5_flores_processed.csv", default_register="formal")
df_d6 = process_anomalous_df("d6_ud_processed.csv", default_register="formal")
df_d7 = process_anomalous_df("d7_custom_augmentation_processed.csv", default_register="informal")

# 1. Merge
print("Concatenating all datasets...")
unified_df = pd.concat([df_d1, df_d2, df_d3, df_d4, df_d5, df_d6, df_d7], ignore_index=True)

# 2. Final Sanity Filter
before_len = len(unified_df)
unified_df = unified_df[unified_df["text"].str.strip() != ""]
unified_df = unified_df.dropna(subset=["text", "iso_code"])
print(f"Dropped {before_len - len(unified_df)} empty/invalid rows after column shifting.")

# 3. Priority Deduplication
# CRITICAL: If the word "ok" exists in D2 (Tatoeba) and D7 (Custom), we want to keep 
# the D7 version because it holds the low_confidence_flag=True metadata.
before_len = len(unified_df)
unified_df = unified_df.sort_values("low_confidence_flag", ascending=False)
unified_df = unified_df.drop_duplicates(subset=["text"], keep="first")
print(f"Removed {before_len - len(unified_df)} cross-corpus duplicates.")

# 4. Global Shuffle
print("Shuffling the unified corpus...")
unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save
os.makedirs(INPUT_DIR, exist_ok=True)
unified_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Unification complete! Final corpus saved to {OUTPUT_FILE} with {len(unified_df):,} rows.")