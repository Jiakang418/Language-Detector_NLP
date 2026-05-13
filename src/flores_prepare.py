from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

language_map = {
    "eng_Latn": ("en", "English"),
    "zho_Hans": ("zh", "Chinese Simplified"),
    "jpn_Jpan": ("ja", "Japanese"),
    "ind_Latn": ("id", "Indonesian"),
    "zsm_Latn": ("ms", "Standard Malay"),
}

pairs = [
    "eng_Latn-zho_Hans",
    "eng_Latn-jpn_Jpan",
    "eng_Latn-ind_Latn",
    "eng_Latn-zsm_Latn",
]

rows = []

for pair in pairs:
    print("Loading:", pair)

    dataset = load_dataset(
        "facebook/flores",
        pair,
        trust_remote_code=True
    )

    for item in dataset["devtest"]:
        for flores_code, (iso_code, language_name) in language_map.items():
            column_name = f"sentence_{flores_code}"

            if column_name in item:
                sentence = item[column_name]

                if sentence:
                    rows.append({
                        "text": sentence.strip(),
                        "iso_code": iso_code,
                        "language_name": language_name,
                        "flores_code": flores_code,
                        "source": "FLORES-200",
                        "usage": "validation"
                    })

df = pd.DataFrame(rows).drop_duplicates()
df.to_csv("data/processed/flores_validation.csv", index=False, encoding="utf-8-sig")

print("Saved: data/processed/flores_validation.csv")
print("Total rows:", len(df))
print(df["iso_code"].value_counts())
print(df.head())