import os
import requests
import pandas as pd
from conllu import parse_incr

os.makedirs("raw_dataset/UD", exist_ok=True)

treebanks = {
    "zh": {
        "language_name": "Chinese",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSD/master/zh_gsd-ud-test.conllu"
    },
    "ja": {
        "language_name": "Japanese",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Japanese-GSD/master/ja_gsd-ud-test.conllu"
    },
    "ko": {
        "language_name": "Korean",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Korean-GSD/master/ko_gsd-ud-test.conllu"
    }
}

rows = []

for iso_code, info in treebanks.items():
    language_name = info["language_name"]
    url = info["url"]
    raw_path = f"raw_dataset/UD/{iso_code}_ud.conllu"

    print(f"Downloading {language_name} UD Treebank...")
    response = requests.get(url)
    response.raise_for_status()

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Extracting sentences from {raw_path}...")

    with open(raw_path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sentence = tokenlist.metadata.get("text")

            if sentence:
                rows.append({
                    "text": sentence.strip(),
                    "iso_code": iso_code,
                    "language_name": language_name,
                    "source": "Universal Dependencies Treebanks",
                    "usage": "preprocessing"
                })

df = pd.DataFrame(rows)
df.to_csv("raw_dataset/UD/ud_cjk_samples.csv", index=False, encoding="utf-8-sig")

print("Saved: raw_dataset/UD/ud_cjk_samples.csv")
print("Total rows:", len(df))
print(df["iso_code"].value_counts())
print(df.head())