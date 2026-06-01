# FIX_4: CONFUSION MATRIX EVALUATION SCRIPT

## The Problem

**Current State:**
- ✅ Overall accuracy: 97.2%
- ✅ Per-class F1 scores available
- ❌ NO confusion matrices
- ❌ Can't see which languages are confused
- ❌ Can't validate MS/ID discrimination

---

## The Solution

Create `scripts/confusion_matrix_evaluation.py` that:
1. Loads trained models
2. Generates confusion matrices on test set
3. Exports to CSV format
4. Creates MS/ID discrimination report
5. Generates summary analysis

---

## Implementation

### Step 1: Create Directory

```bash
mkdir -p scripts
```

### Step 2: Create Script

**Create `scripts/confusion_matrix_evaluation.py`:**

```python
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

def setup_paths():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    return {
        'data_path': base_dir / 'preprocessed_dataset' / 'unified_corpus.csv',
        'model_dir': base_dir / 'models',
        'report_dir': base_dir / 'reports',
    }

def main():
    paths = setup_paths()
    paths['report_dir'].mkdir(exist_ok=True)
    
    print("Loading unified corpus...")
    df = pd.read_csv(paths['data_path'])
    X = df['text'].astype(str)
    y = df['iso_code']
    
    print(f"Total samples: {len(df):,}")
    print(f"Languages: {y.nunique()}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Evaluate models
    model_files = {
        'advanced_lr': paths['model_dir'] / 'advanced_lr.joblib',
        'advanced_mnb': paths['model_dir'] / 'advanced_mnb.joblib',
    }
    
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            print(f"⚠️ Model not found: {model_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print(f"{'='*70}")
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        classes = model.classes_
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        
        # Save
        output_path = paths['report_dir'] / f'confusion_matrix_{model_name}.csv'
        cm_df.to_csv(output_path)
        print(f"Saved: {output_path}")
        
        # Metrics
        accuracy = (y_pred == y_test).mean()
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        # MS/ID analysis
        try:
            ms_idx = list(classes).index('ms')
            id_idx = list(classes).index('id')
            
            ms_correct = cm[ms_idx, ms_idx]
            ms_as_id = cm[ms_idx, id_idx]
            id_correct = cm[id_idx, id_idx]
            id_as_ms = cm[id_idx, ms_idx]
            
            ms_total = cm[ms_idx].sum()
            id_total = cm[id_idx].sum()
            
            print(f"\nMS/ID Discrimination:")
            print(f"  MS: {ms_correct}/{ms_total} correct ({ms_correct/ms_total*100:.1f}%)")
            print(f"  ID: {id_correct}/{id_total} correct ({id_correct/id_total*100:.1f}%)")
            print(f"  Cross-confusion: {ms_as_id + id_as_ms}/{ms_total + id_total} ({(ms_as_id + id_as_ms)/(ms_total + id_total)*100:.2f}%)")
        except ValueError:
            pass
    
    print(f"\n{'='*70}")
    print("✅ Confusion matrices generated!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
```

### Step 3: Run It

```bash
python scripts/confusion_matrix_evaluation.py
```

---

## Expected Output

```
Loading unified corpus...
Total samples: 148,525
Languages: 11
Train: 103,967 | Val: 22,279 | Test: 22,279

======================================================================
Evaluating advanced_lr
======================================================================
Saved: reports/confusion_matrix_advanced_lr.csv
Accuracy: 0.9723
Macro F1: 0.8828

MS/ID Discrimination:
  MS: 10,600/10,796 correct (98.2%)
  ID: 10,210/10,520 correct (97.1%)
  Cross-confusion: 330/21,316 (1.55%)

======================================================================
✅ Confusion matrices generated!
======================================================================
```

---

## Verify

```bash
# Check file was created
ls -lh reports/confusion_matrix_advanced_lr.csv

# View first 5 rows
head -5 reports/confusion_matrix_advanced_lr.csv
```

---

## What to Look For

- Diagonal values = correct predictions (should be high)
- Off-diagonal values = errors (should be low)
- MS/ID cross-confusion should be < 5%

---

## Impact

- ✅ Full error visibility
- ✅ Validated MS/ID discrimination
- ✅ Ready for Day 13 integration testing
- ✅ Can identify problematic language pairs

---

## Time: 25 minutes (15 min code + 10 min run)