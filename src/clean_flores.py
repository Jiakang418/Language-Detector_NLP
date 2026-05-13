import pandas as pd
from preprocess import preprocess_text, detect_script

input_path = "data/processed/flores_validation.csv"
output_path = "data/processed/flores_validation_cleaned.csv"

df = pd.read_csv(input_path)

df["detected_script"] = df["text"].apply(detect_script)
df["cleaned_text"] = df["text"].apply(preprocess_text)

df = df[df["cleaned_text"].str.strip() != ""]

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Saved:", output_path)
print("Original rows:", len(pd.read_csv(input_path)))
print("Cleaned rows:", len(df))
print()
print("Detected script counts:")
print(df["detected_script"].value_counts())
print()
print(df[["text", "cleaned_text", "iso_code", "detected_script"]].head(10))