# D5, D6, and D7 Data Cleaning & Preprocessing Summary

## 1. Overview

This document explains the data cleaning and preprocessing work completed for Dataset 5, Dataset 6, and Dataset 7 in the Language Detector project.

The assigned datasets are:

| Dataset | Dataset Name | Main Function |
|---|---|---|
| D5 | FLORES-200 | Held-out validation dataset for multilingual evaluation and confidence scoring |
| D6 | Universal Dependencies Treebanks | Script-aware preprocessing support for Chinese, Japanese, and Korean text |
| D7 | Custom Augmentation Dataset | Adversarial testing for short, ambiguous, mixed, and low-confidence inputs |

The main goal of this section is to prepare clean, structured, and usable data for the language detection pipeline.

This preprocessing work supports the masterplan’s **Node 2: Preprocessing Gate**, where raw text is normalised, cleaned, script-detected, and prepared before feature extraction. D5 also supports **Node 5: Confidence Scoring**, while D7 supports low-confidence boundary testing and continuous validation.

---

## 2. Role of Preprocessing in the Language Detector Pipeline

Before the model can classify text into a language, the raw input must be cleaned and standardised.

Raw user input may contain:

- uppercase and lowercase letters,
- emojis,
- numbers,
- punctuation,
- inconsistent spacing,
- non-Latin characters,
- mixed scripts,
- very short ambiguous words,
- symbol-only inputs.

If these issues are not handled before feature extraction, the classifier may learn noisy or inconsistent patterns.

Therefore, preprocessing is important because it prepares the text into a cleaner format before it is passed into the TF-IDF character n-gram vectoriser and the classification model.

The preprocessing stage helps:

1. reduce noise,
2. standardise text,
3. detect writing systems,
4. preserve useful language patterns,
5. handle non-Latin scripts,
6. improve model robustness,
7. support confidence scoring.

---

## 3. Summary of Completed Work

The following major tasks were completed:

| Task | Status |
|---|---|
| Created project folder structure | Completed |
| Created Python virtual environment | Completed |
| Installed required libraries | Completed |
| Prepared actual FLORES-200 validation dataset | Completed |
| Cleaned FLORES-200 dataset | Completed |
| Generated FLORES-200 quality report | Completed |
| Downloaded actual UD Treebank files | Completed |
| Extracted Chinese, Japanese, and Korean UD samples | Completed |
| Cleaned UD Treebank samples | Completed |
| Generated UD quality report | Completed |
| Created custom adversarial D7 dataset | Completed |
| Cleaned D7 dataset | Completed |
| Improved Korean script detection for `ㅋㅋㅋ` | Completed |
| Generated D7 quality report | Completed |
| Built shared preprocessing pipeline | Completed |

---

# 4. D5: FLORES-200

## 4.1 Dataset Purpose

FLORES-200 is used as a multilingual validation dataset.

In this project, FLORES-200 is not used as the main training dataset. Instead, it acts as a held-out validation set. This means the dataset is kept separate from the main training data and is used later to test whether the model can generalise to unseen multilingual examples.

D5 supports:

- multilingual validation,
- confidence score evaluation,
- rare-language evaluation,
- external benchmark testing,
- reliability checking after model training.

This is important because a model may perform well on its training dataset but fail on unseen data. FLORES-200 helps test whether the language detector performs well on independent multilingual text.

---

## 4.2 Why FLORES-200 Is Useful

FLORES-200 is useful because it contains multilingual sentence data across many languages. In this project, it helps test whether the model can correctly identify language labels for different scripts and language families.

For example, it supports evaluation of:

- English text,
- Malay text,
- Indonesian text,
- Chinese text,
- Japanese text.

This aligns well with the project’s focus on multilingual language detection and script diversity.

---

## 4.3 Languages Prepared from FLORES-200

The prepared FLORES-200 subset includes:

| Language | ISO Code | FLORES Code |
|---|---|---|
| English | en | eng_Latn |
| Chinese Simplified | zh | zho_Hans |
| Japanese | ja | jpn_Jpan |
| Indonesian | id | ind_Latn |
| Standard Malay | ms | zsm_Latn |

These languages were selected because they are directly relevant to the project’s main challenges:

| Challenge | Relevant Dataset Language |
|---|---|
| Latin-script detection | English, Malay, Indonesian |
| Malay vs. Indonesian similarity | Malay, Indonesian |
| CJK script handling | Chinese, Japanese |
| Multilingual validation | All selected languages |

---

## 4.4 FLORES-200 Preparation Script

The preparation script used for D5 is:

```text
src/flores_prepare.py