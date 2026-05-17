import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time
import os
from pathlib import Path

# Setup paths relative to script location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "preprocessed_dataset" / "unified_corpus.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    print("Loading unified dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Check for NaN and remove
    df = df.dropna(subset=['text', 'iso_code'])
    
    X = df['text']
    y = df['iso_code']
    
    print(f"Dataset loaded. Total samples: {len(df)}. Total languages: {y.nunique()}")
    
    print("Splitting dataset (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Configuring Hybrid Feature Extraction...")
    # Word-level TF-IDF (1-2 ngrams) to capture vocabulary and common phrases
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=15000
    )
    
    # Character-level TF-IDF (2-4 ngrams) to capture morphology, subwords, and script properties
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=15000
    )
    
    # Combine both feature sets
    hybrid_features = FeatureUnion([
        ('word_tfidf', word_vectorizer),
        ('char_tfidf', char_vectorizer)
    ])
    
    # Define baseline models as per the proposal
    models = {
        'Multinomial_Naive_Bayes': MultinomialNB(),
        'Logistic_Regression': LogisticRegression(max_iter=300, n_jobs=-1, multi_class='ovr') 
        # multi_class='ovr' with n_jobs=-1 allows parallel training of classifiers
    }
    
    report_lines = []
    report_lines.append("Baseline Models Evaluation Report\n")
    report_lines.append("="*60 + "\n")
    
    for model_name, classifier in models.items():
        print(f"\n--- Training {model_name} ---")
        pipeline = Pipeline([
            ('features', hybrid_features),
            ('classifier', classifier)
        ])
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")
        
        print(f"Evaluating {model_name}...")
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        eval_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        
        report_lines.append(f"Model: {model_name}\n")
        report_lines.append(f"Training Time: {train_time:.2f} seconds\n")
        report_lines.append(f"Inference Time: {eval_time:.2f} seconds (on {len(y_test)} samples)\n")
        report_lines.append(f"Accuracy: {acc:.4f}\n")
        report_lines.append("Classification Report:\n")
        report_lines.append(class_report + "\n")
        report_lines.append("-" * 60 + "\n")
        
        # Save model
        model_path = MODEL_DIR / f"baseline_{model_name.lower()}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_name} to {model_path}")
        
    # Write report
    report_path = REPORT_DIR / "baseline_metrics.txt"
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
        
    print(f"\nProcess complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()
