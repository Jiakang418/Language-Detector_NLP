"""
=============================================================================
D3 — CC-100 Corpus (Malay + Indonesian Subsets) Pipeline
=============================================================================
Source      : https://huggingface.co/datasets/cc100
              Subsets: 'ms' (Malay) and 'id' (Indonesian)
ISO Scheme  : ISO 639-1 — 'ms' and 'id' (already correct in CC-100)
Register    : mixed (web crawl: news, blogs, forums, product listings)
Script      : latin (both languages use Latin script exclusively)
Key Role    : Cross-lingual Malay vs. Indonesian disambiguation.
              Large-volume, naturalistic web text for both languages.
              Also produces ms_id_anchors.json — discriminating N-gram
              features injected into the vectoriser vocabulary later.
Output      : preprocessed_dataset/d3_cc100_processed.csv
              preprocessed_dataset/ms_id_anchors.json
=============================================================================
"""

import os
import re
import json
import unicodedata
import urllib.request
import lzma

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# PATHS  — resolved relative to this script file so the pipeline works
#          regardless of the working directory it is launched from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

OUTPUT_DIR   = os.path.join(_PROJECT_ROOT, "preprocessed_dataset")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "d3_cc100_processed.csv")
ANCHORS_FILE = os.path.join(OUTPUT_DIR, "ms_id_anchors.json")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Sample size per language — CC-100 is enormous; sample strategically.
# We stream and oversample by 3x to absorb cleaning losses.
SAMPLE_SIZE  = 50_000
MIN_TOKENS   = 3    # Minimum whitespace-token count after cleaning
MAX_TOKENS   = 80   # Drop very long sentences (OCR artifacts, SEO spam)

# N-gram anchor extraction
ANCHOR_NGRAM_RANGE = (2, 4)   # Character N-gram range for discriminator extraction
TOP_N_ANCHORS      = 200      # Top N discriminating N-grams per language

TARGET_LANGUAGES_ISO1 = {
    "ms": "Malay",
    "id": "Indonesian",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """
    Detect the dominant script via Unicode block inspection.
    Both ms and id are exclusively Latin, but we run detection anyway
    for schema uniformity with the other dataset pipelines.
    Returns one of: 'latin', 'cjk', 'arabic', 'cyrillic', 'other'.
    """
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


def count_tokens(text: str, script: str = "latin") -> int:
    """
    Word count for Latin/Arabic/Cyrillic (whitespace split).
    Character count for CJK (no natural word boundaries).
    """
    if script == "cjk":
        return len([ch for ch in text if not ch.isspace()])
    return len(text.split())


# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------

def clean_text_cc100(text: str) -> str:
    """
    CC-100-specific cleaning — more aggressive than other pipelines
    because web crawl data contains URLs, HTML artifacts, SEO chaff, etc.

    Pipeline (order matters):
      1. Remove URLs
      2. Remove email addresses
      3. Remove HTML entity remnants  (&amp; &nbsp; &#39; etc.)
      4. Reject lines that are >50% non-alphabetic  (nav menus, SEO spam)
      5. Lowercase
      6. Remove digits
      7. Remove emojis and Unicode control/format characters
      8. Collapse whitespace and strip
    """
    # 1. URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 2. Email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # 3. HTML entities
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)

    # 4. Punctuation-heavy lines (nav menus, structured data, SEO lists)
    chars = [c for c in text if c.strip()]
    if chars:
        punct_ratio = sum(1 for c in chars if not c.isalpha()) / len(chars)
        if punct_ratio > 0.5:
            return ""

    # 5. Lowercase
    text = text.lower()

    # 6. Digits
    text = re.sub(r"\d+", "", text)

    # 7. Emojis and Unicode control/format/symbol categories
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Sm", "Sk", "Cc", "Cf")
    )

    # 8. Whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_cc100_subset(lang: str, sample_size: int) -> pd.DataFrame:
    """
    Stream a single CC-100 language subset directly from the source server.
    Bypasses the Hugging Face datasets library entirely due to deprecation
    of custom dataset scripts.

    Args:
        lang        : ISO 639-1 code — 'ms' or 'id'
        sample_size : Target number of rows after cleaning
    """
    url = f"https://data.statmt.org/cc-100/{lang}.txt.xz"
    print(f"\n[D3] Streaming CC-100 subset directly from: {url}")
    print(f"[D3] Target: {sample_size:,} rows (oversampling x3)...")

    rows = []
    oversample_target = sample_size * 3

    try:
        # Open a network stream directly to the .xz file
        with urllib.request.urlopen(url) as response:
            # Wrap the network stream in Python's native LZMA decompressor
            with lzma.open(response, mode='rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= oversample_target:
                        break
                    
                    text = line.strip()
                    tokens = text.split()
                    
                    # Cheap pre-filter
                    if MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
                        rows.append(text)
                        
                    # Print progress every 50k lines read so you know it hasn't frozen
                    if i > 0 and i % 50_000 == 0:
                        print(f"[D3]   ...scanned {i:,} lines directly over network", end="\r")

        print() # Clear the carriage return line
    except Exception as e:
        raise RuntimeError(f"[D3] Failed to stream from {url}. Error: {e}")

    df = pd.DataFrame({"text": rows, "iso_code": lang})
    print(f"[D3]   Raw candidate rows ({lang}): {len(df):,}")
    return df
# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------

def transform_cc100(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline for the combined ms + id raw DataFrame.

    Steps:
      1. Clean text (CC-100 aggressive mode)
      2. Detect script
      3. Count tokens
      4. Quality filter  (empty text, token range)
      5. Deduplicate
      6. Sample to SAMPLE_SIZE per language
      7. Attach unified-schema metadata columns
    """

    # ---- Step 1: Clean ----
    print("[D3] Cleaning text (CC-100 mode)...")
    df["text"] = df["text"].astype(str).apply(clean_text_cc100)

    # ---- Step 2: Script ----
    print("[D3] Detecting scripts...")
    df["script"] = df["text"].apply(detect_script)

    # ---- Step 3: Token count ----
    df["token_count"] = df.apply(
        lambda row: count_tokens(row["text"], row["script"]), axis=1
    )

    # ---- Step 4: Quality filter ----
    before = len(df)
    df = df[df["text"].str.strip().ne("")]
    df = df[df["token_count"].between(MIN_TOKENS, MAX_TOKENS)]
    print(f"[D3] Rows dropped by quality filter: {before - len(df):,}")

    # ---- Step 5: Deduplicate ----
    before = len(df)
    df.drop_duplicates(subset="text", inplace=True)
    print(f"[D3] Duplicates removed: {before - len(df):,}")

    # ---- Step 6: Sample per language ----
    df = (
        df.groupby("iso_code", group_keys=False)
          .apply(lambda g: g.sample(
              n=min(len(g), SAMPLE_SIZE),
              random_state=RANDOM_SEED
          ))
    )
    print(f"[D3] Rows after sampling: {len(df):,}")

    # ---- Step 7: Metadata ----
    df["language_name"] = df["iso_code"].map(TARGET_LANGUAGES_ISO1)
    df["source"]        = "cc100"
    df["register"]      = "mixed"

    # Reorder to unified schema column order
    return df[[
        "text", "iso_code", "language_name",
        "source", "register", "token_count", "script"
    ]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# N-GRAM ANCHOR EXTRACTION
# ---------------------------------------------------------------------------

def extract_ms_id_anchors(df: pd.DataFrame) -> dict:
    """
    Extract character N-grams that statistically discriminate Malay from
    Indonesian. These anchors are saved to ms_id_anchors.json and later
    injected into the TF-IDF vectoriser's forced vocabulary to guarantee
    the model retains discriminating signals even for very short inputs.

    Method:
      1. Compute per-N-gram frequency sums for the ms corpus
      2. Compute per-N-gram frequency sums for the id corpus
      3. Discrimination score per N-gram:
             score_ms = freq_ms / (freq_ms + freq_id + 1)
             score_id = freq_id / (freq_ms + freq_id + 1)
      4. Keep the top TOP_N_ANCHORS for each language

    Linguistic ground truth (verify top anchors against these):
      Malay      : 'kan', 'lah', 'kah', 'ber', 'ke'
      Indonesian : 'nya', 'me', 'di', 'kan' (less dominant than ms)
    """
    print("\n[D3] Extracting Malay/Indonesian N-gram anchors...")

    df_ms = df[df["iso_code"] == "ms"]["text"].reset_index(drop=True)
    df_id = df[df["iso_code"] == "id"]["text"].reset_index(drop=True)
    n_ms  = len(df_ms)

    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=ANCHOR_NGRAM_RANGE,
        max_features=50_000,
        min_df=10,    # N-gram must appear in at least 10 documents
    )

    # Fit on the combined ms + id corpus to build a shared vocabulary
    combined = pd.concat([df_ms, df_id], ignore_index=True)
    X = vectorizer.fit_transform(combined)

    # Split back into per-language frequency matrices
    X_ms = X[:n_ms]
    X_id = X[n_ms:]

    feature_names = vectorizer.get_feature_names_out()

    freq_ms = pd.Series(X_ms.sum(axis=0).A1, index=feature_names)
    freq_id = pd.Series(X_id.sum(axis=0).A1, index=feature_names)

    # Discrimination scores
    score_ms = freq_ms / (freq_ms + freq_id + 1)
    score_id = freq_id / (freq_ms + freq_id + 1)

    top_ms = score_ms.nlargest(TOP_N_ANCHORS).index.tolist()
    top_id = score_id.nlargest(TOP_N_ANCHORS).index.tolist()

    anchors = {
        "ms_anchors": top_ms,
        "id_anchors": top_id,
        "metadata": {
            "ngram_range":    list(ANCHOR_NGRAM_RANGE),
            "top_n":          TOP_N_ANCHORS,
            "ms_corpus_size": int(n_ms),
            "id_corpus_size": int(len(df_id)),
        }
    }

    print(f"[D3] Top 10 Malay anchors     : {top_ms[:10]}")
    print(f"[D3] Top 10 Indonesian anchors: {top_id[:10]}")
    return anchors


def save_anchors(anchors: dict) -> None:
    """Persist the anchor dict to preprocessed_dataset/ms_id_anchors.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ANCHORS_FILE, "w", encoding="utf-8") as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    print(f"[D3] Anchors saved  -> {ANCHORS_FILE}")


# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------

def validate_cc100(df: pd.DataFrame) -> None:
    """
    Assert output conforms to the unified schema.
    Raises AssertionError with a descriptive message on any violation
    so the pipeline halts before writing a bad CSV.
    """
    print("[D3] Validating output...")

    # Only ms and id should be present
    unexpected = set(df["iso_code"].unique()) - {"ms", "id"}
    assert not unexpected, f"Unexpected iso_codes: {unexpected}"

    # No nulls in any schema column
    for col in ["text", "iso_code", "language_name", "source", "register", "script"]:
        nulls = df[col].isna().sum()
        assert nulls == 0, f"Null values in column '{col}': {nulls}"

    # Register must be 'mixed' for all CC-100 rows
    assert (df["register"] == "mixed").all(), \
        "Unexpected register value in CC-100 data — expected 'mixed' for all rows"

    # Source must be 'cc100'
    assert (df["source"] == "cc100").all(), \
        "Unexpected source value — expected 'cc100' for all rows"

    # Both languages must be present
    counts = df["iso_code"].value_counts()
    assert "ms" in counts.index, "Malay (ms) is missing from output"
    assert "id" in counts.index, "Indonesian (id) is missing from output"

    # Warn if class balance is poor (>2x imbalance between ms and id)
    ratio = counts.max() / counts.min()
    if ratio > 2.0:
        print(f"[D3] WARNING: ms/id imbalance ratio = {ratio:.2f} (threshold 2.0)")

    # No within-language duplicate texts
    for iso in ["ms", "id"]:
        subset = df[df["iso_code"] == iso]
        dups = subset.duplicated(subset="text").sum()
        assert dups == 0, f"Duplicate texts in {iso} subset: {dups}"

    # Token count bounds
    assert df["token_count"].ge(MIN_TOKENS).all(), \
        f"token_count below minimum ({MIN_TOKENS}) found"
    assert df["token_count"].le(MAX_TOKENS).all(), \
        f"token_count above maximum ({MAX_TOKENS}) found"

    print("[D3] Validation PASSED ✓")
    print(f"[D3] Language distribution:\n{df['iso_code'].value_counts().to_string()}")
    print(f"[D3] Token count stats:\n{df['token_count'].describe().to_string()}\n")


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def save_cc100(df: pd.DataFrame) -> None:
    """Write the processed DataFrame to preprocessed_dataset/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[D3] Saved -> {OUTPUT_FILE} ({len(df):,} rows)\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    """
    Full D3 pipeline:
      1. Stream ms and id subsets from CC-100 (HuggingFace)
      2. Clean, filter, deduplicate, sample
      3. Extract and save Malay/Indonesian N-gram anchors
      4. Validate schema
      5. Save to preprocessed_dataset/d3_cc100_processed.csv
    """
    print("=" * 65)
    print("D3 — CC-100 Malay + Indonesian Preprocessing Pipeline")
    print("=" * 65)

    # Load both language subsets from HuggingFace streaming
    df_ms  = load_cc100_subset("ms", SAMPLE_SIZE)
    df_id  = load_cc100_subset("id", SAMPLE_SIZE)
    df_raw = pd.concat([df_ms, df_id], ignore_index=True)

    # Transform to unified schema
    df_processed = transform_cc100(df_raw)

    # Extract N-gram discriminators (side output — used by vectoriser later)
    anchors = extract_ms_id_anchors(df_processed)
    save_anchors(anchors)

    # Validate then save
    validate_cc100(df_processed)
    save_cc100(df_processed)

    return df_processed


if __name__ == "__main__":
    run()