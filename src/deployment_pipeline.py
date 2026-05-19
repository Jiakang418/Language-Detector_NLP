import joblib
import time
import os
import numpy as np
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
# We will use the Advanced Logistic Regression model since it achieved the highest accuracy (97.23%)
MODEL_PATH = BASE_DIR / "models" / "advanced_lr.joblib"

class LanguageDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Loading champion model from {model_path}...")
        start_time = time.time()
        self.model = joblib.load(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.3f} seconds.")
        
    def predict(self, text):
        """
        Returns the predicted ISO language code and the statistical confidence percentage.
        Addresses Probability Estimation and I/O Handoff for Gradio.
        """
        # Ensure input is a list-like structure for the pipeline
        if isinstance(text, str):
            text = [text]
            
        start_time = time.time()
        
        # 1. Probability Estimation: Get probabilities for all classes
        probabilities = self.model.predict_proba(text)
        
        # 2. Extract the index of the highest probability
        best_indices = probabilities.argmax(axis=1)
        
        # 3. Retrieve the actual class label (ISO code)
        predicted_classes = self.model.classes_[best_indices]
        
        # 4. Calculate the confidence score percentage
        confidence_scores = np.max(probabilities, axis=1) * 100
        
        # System Latency Profiling
        latency_ms = (time.time() - start_time) * 1000 
        
        results = []
        for i in range(len(text)):
            results.append({
                "text": text[i],
                "predicted_iso_code": predicted_classes[i],
                "confidence_score": f"{round(confidence_scores[i], 2)}%",
                "latency_ms": round(latency_ms / len(text), 2)
            })
            
        # Return single dict if input was single string, else list of dicts
        return results[0] if len(results) == 1 else results

if __name__ == "__main__":
    print("--- Model Evaluation, Confidence Calibration & Latency Profiling ---\n")
    detector = LanguageDetector()
    
    # Test cases representing the difficult scenarios
    test_texts = [
        "Halo", # Linguistic Ambiguity (Short text)
        "Saya pergi ke pasar pagi ini.", # Malay / Indonesian Similarity
        "The quick brown fox jumps over the lazy dog.", # English
        "Bonjour, comment allez-vous aujourd'hui?", # French
        "今天天气很好", # Script Diversity (Chinese Logographic)
        "오늘은 날씨가 좋네요" # Korean
    ]
    
    print("\n--- Running Inference Tests ---")
    for txt in test_texts:
        result = detector.predict(txt)
        print(f"Text: '{txt}'")
        print(f"  -> Predicted: {result['predicted_iso_code']}")
        print(f"  -> Confidence: {result['confidence_score']}")
        print(f"  -> Speed: {result['latency_ms']} ms\n")
