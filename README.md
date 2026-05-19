# 🌐 Language Detector (NLP)

A lightweight, high-performance Natural Language Processing application designed to instantly identify the language of any given text. Built with a unified corpus of over 250,000 samples, this tool serves as a foundational gateway for downstream NLP tasks such as automated translation, localized content filtering, and sentiment analysis.

## ✨ Features

- **Blazing Fast Inference:** Achieves ~3ms latency per prediction, making it perfectly suited for real-time applications.
- **High Accuracy (97.2%):** Powered by an advanced Logistic Regression classifier using highly optimized TF-IDF FeatureUnions.
- **Cross-Lingual Resolution:** Expertly distinguishes between highly similar languages (like Malay `ms` and Indonesian `id`) by dynamically weighting language-specific anchor n-grams.
- **Script Diversity Handling:** Natively parses CJK (Chinese, Japanese, Korean) logographic scripts alongside Latin text using Character N-gram Tokenization.
- **Ambiguity Detection:** Outputs a statistical **Confidence Score** for every prediction. Highly ambiguous or short text (e.g., "Halo") receives appropriate probability calibration.
- **Beautiful UI:** Comes with a fully interactive Web UI powered by Gradio.

## 🗂 Supported Languages (11 Classes)

- 🇬🇧 English (`en`)
- 🇲🇾 Malay (`ms`)
- 🇮🇩 Indonesian (`id`)
- 🇨🇳 Chinese (`zh`)
- 🇯🇵 Japanese (`ja`)
- 🇰🇷 Korean (`ko`)
- 🇫🇷 French (`fr`)
- 🇩🇪 German (`de`)
- 🇪🇸 Spanish (`es`)
- 🇸🇦 Arabic (`ar`)
- ❓ Unknown (`unknown`)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Install Dependencies
Open your terminal (or Command Prompt / PowerShell) in the root folder of this project and install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Once the dependencies are installed, you can start the Gradio UI by running:

```bash
python app.py
```

After a few seconds, the terminal will provide a local web link (usually `http://127.0.0.1:7860`). `Ctrl+Click` that link to open the Language Detector interface in your browser!

---

## 🧠 Model Architecture

The underlying architecture relies on a **Scikit-Learn Pipeline** that processes raw text directly into predictions:
1. **Word-Level TF-IDF (1-2 ngrams):** Captures broad vocabulary and common phrases.
2. **Character-Level TF-IDF (2-4 ngrams):** Captures morphology, subwords, and logographic scripts (Chinese).
3. **Anchor Vectorizer:** A targeted character vectorizer explicitly pre-loaded with cross-lingual discriminators to break ties between Malay and Indonesian.
4. **Classifier:** A class-weight-balanced `LogisticRegression(multi_class='ovr')` model.

*Note: You can view or retrain the models in the `src/` directory using `python src/advanced_training.py`.*
