import pandas as pd

input_path = "data/processed/ud_cjk_cleaned.csv"
report_path = "reports/ud_quality_report.txt"

df = pd.read_csv(input_path)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Universal Dependencies Treebanks Data Cleaning Quality Report\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Total cleaned rows: {len(df)}\n")
    f.write(f"Total columns: {len(df.columns)}\n\n")

    f.write("Rows per language:\n")
    f.write(str(df["iso_code"].value_counts()))
    f.write("\n\n")

    f.write("Detected script counts:\n")
    f.write(str(df["detected_script"].value_counts()))
    f.write("\n\n")

    f.write("Average original text length:\n")
    f.write(str(df["text"].astype(str).str.len().mean()))
    f.write("\n\n")

    f.write("Average cleaned text length:\n")
    f.write(str(df["cleaned_text"].astype(str).str.len().mean()))
    f.write("\n\n")

    empty_count = df["cleaned_text"].isna().sum() + (
        df["cleaned_text"].astype(str).str.strip() == ""
    ).sum()

    f.write(f"Empty cleaned rows: {empty_count}\n\n")

    f.write("Sample cleaned rows:\n")
    f.write(str(df[["text", "cleaned_text", "iso_code", "detected_script"]].head(10)))

print("Saved:", report_path)