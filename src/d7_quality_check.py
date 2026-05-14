import pandas as pd

input_path = "preprocessed_dataset/d7_custom_augmentation_processed.csv"
report_path = "reports/d7_quality_report.txt"

df = pd.read_csv(input_path)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("D7 Custom Augmentation Dataset Quality Report\n")
    f.write("=" * 55 + "\n\n")

    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Total columns: {len(df.columns)}\n\n")

    f.write("Rows by ISO code:\n")
    f.write(str(df["iso_code"].value_counts()))
    f.write("\n\n")

    f.write("Low confidence flag distribution:\n")
    f.write(str(df["low_confidence_flag"].value_counts()))
    f.write("\n\n")

    f.write("Detected script counts:\n")
    f.write(str(df["detected_script"].value_counts()))
    f.write("\n\n")

    empty_count = (
        df["cleaned_text"].isna().sum()
        + (df["cleaned_text"].astype(str).str.strip() == "").sum()
    )

    f.write(f"Empty cleaned rows: {empty_count}\n\n")

    f.write("Average original text length:\n")
    f.write(str(df["text"].astype(str).str.len().mean()))
    f.write("\n\n")

    f.write("Average cleaned text length:\n")
    f.write(str(df["cleaned_text"].astype(str).str.len().mean()))
    f.write("\n\n")

    f.write("Sample cleaned rows:\n")
    f.write(str(df[[
        "text",
        "cleaned_text",
        "iso_code",
        "detected_script",
        "low_confidence_flag"
    ]].head(24)))

print("Saved:", report_path)