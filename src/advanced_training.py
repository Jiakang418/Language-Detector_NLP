import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "preprocessed_dataset" / "unified_corpus.csv"
ANCHORS_PATH = BASE_DIR / "preprocessed_dataset" / "ms_id_anchors.json"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    print("Loading dataset and anchor keywords...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text', 'iso_code'])
    
    # 1. Addressing Cross-Lingual Similarity (Malay vs Indonesian)
    # We load the carefully extracted anchors to explicitly force the model to 
    # weight these distinguishing N-grams during Feature Extraction.
    with open(ANCHORS_PATH, 'r') as f:
        anchors_data = json.load(f)
        
    ms_anchors = anchors_data.get('ms_anchors', [])
    id_anchors = anchors_data.get('id_anchors', [])
    combined_anchors = list(set(ms_anchors + id_anchors))
    print(f"Loaded {len(combined_anchors)} MS/ID specific anchor n-grams.")

    X = df['text']
    y = df['iso_code']
    
    # Subsample for Hyperparameter Tuning (10% of data ~25k samples) to keep tuning time reasonable
    _, X_tune, _, y_tune = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    # Full split for final training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Configuring Advanced Feature Extraction...")
    
    # 2. Addressing Linguistic Ambiguity (Short texts)
    # Word n-grams help with context, but we need character n-grams to catch 
    # sub-words in very short texts like "Halo" or "Ok".
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=15000)
    
    # 3. Script Diversity Handling (Chinese Logographics vs Latin)
    # The character n-gram analyzer inherently splits Chinese text into characters properly,
    # completely bypassing the need for whitespace tokenization.
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=15000)
    
    # Explicit Anchor Vectorizer for Malay/Indo specific separation
    anchor_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), vocabulary=combined_anchors)
    
    advanced_features = FeatureUnion([
        ('word_tfidf', word_vectorizer),
        ('char_tfidf', char_vectorizer),
        ('anchor_tfidf', anchor_vectorizer) # Guaranteed features for closely related languages
    ])
    
    # 4. Hyperparameter Tuning using RandomizedSearchCV
    print("\nStarting Hyperparameter Tuning on subsample (this may take a minute)...")
    
    # Tuning Multinomial Naive Bayes
    mnb_pipeline = Pipeline([
        ('features', advanced_features),
        ('clf', MultinomialNB())
    ])
    
    mnb_params = {
        'clf__alpha': [0.1, 0.5, 1.0],
        'features__char_tfidf__ngram_range': [(2, 3), (2, 4)]
    }
    
    mnb_search = RandomizedSearchCV(mnb_pipeline, mnb_params, n_iter=3, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    print("Tuning Multinomial Naive Bayes parameters...")
    mnb_search.fit(X_tune, y_tune)
    best_mnb = mnb_search.best_estimator_
    print(f"Best MNB Params: {mnb_search.best_params_}")
    
    # Tuning Logistic Regression
    lr_pipeline = Pipeline([
        ('features', advanced_features),
        ('clf', LogisticRegression(max_iter=300, multi_class='ovr', solver='lbfgs', class_weight='balanced'))
    ])
    
    lr_params = {
        'clf__C': [0.5, 1.0, 2.0],
        'features__word_tfidf__max_features': [10000, 15000]
    }
    
    lr_search = RandomizedSearchCV(lr_pipeline, lr_params, n_iter=2, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    print("Tuning Logistic Regression parameters...")
    lr_search.fit(X_tune, y_tune)
    best_lr = lr_search.best_estimator_
    print(f"Best LR Params: {lr_search.best_params_}")
    
    models = {
        'Advanced_MNB': best_mnb,
        'Advanced_LR': best_lr
    }
    
    report_lines = ["Advanced Models Evaluation Report\n", "="*60 + "\n"]
    report_lines.append(f"Best MNB Params: {mnb_search.best_params_}\n")
    report_lines.append(f"Best LR Params: {lr_search.best_params_}\n\n")
    
    for model_name, pipeline in models.items():
        print(f"\n--- Retraining {model_name} on Full Dataset ---")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Evaluating {model_name}...")
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        report_lines.append(f"Model: {model_name}\n")
        report_lines.append(f"Training Time: {train_time:.2f} seconds\n")
        report_lines.append(f"Accuracy: {acc:.4f}\n")
        report_lines.append("Classification Report:\n")
        report_lines.append(class_report + "\n")
        report_lines.append("-" * 60 + "\n")
        
        # Save advanced model
        model_path = MODEL_DIR / f"{model_name.lower()}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_name} to {model_path}")
        
    # Write report
    report_path = REPORT_DIR / "advanced_metrics.txt"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
        
    print(f"\nAdvanced Training Complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()
