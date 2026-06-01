# IMPLEMENTATION_GUIDE_DAY11_FIXES: Step-by-Step Execution

## ⏱️ Timeline Overview

```
PHASE 1-4: Code Patches              30 min  (manual)
PHASE 5-6: Setup & Corpus            10 min  (automatic)
PHASE 7:   Retrain Models          4-6 hrs  (automatic ☕)
PHASE 8:   Generate Evaluation      15 min  (automatic)
PHASE 9:   Validation                30 min  (manual)
─────────────────────────────────────────────
TOTAL:                            5-7 HOURS
```

---

## 📋 Phase 1-4: Code Patches (30 Minutes)

### Phase 1: Update requirements.txt (5 min)

```bash
cat > requirements.txt << 'EOF'
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0,<1.7.0
joblib>=1.3.0
gradio>=4.0.0,<5.0.0
ftfy>=6.0.0
conllu>=4.0.0
requests>=2.25.0
EOF

# Verify
cat requirements.txt
```

---

### Phase 2: Patch src/baseline_training.py (5 min)

```bash
# Find and replace using sed
sed -i "s/multi_class='ovr'/multi_class='multinomial', solver='lbfgs'/g" src/baseline_training.py

# Or manually:
# Find line 62: LogisticRegression(max_iter=300, n_jobs=-1, multi_class='ovr')
# Replace with: LogisticRegression(max_iter=300, n_jobs=-1, multi_class='multinomial', solver='lbfgs')
```

---

### Phase 3: Patch src/advanced_training.py (5 min)

```bash
# Find and replace using sed
sed -i "s/multi_class='ovr'/multi_class='multinomial'/g" src/advanced_training.py

# Or manually:
# Find line 94: LogisticRegression(max_iter=300, multi_class='ovr', solver='lbfgs', class_weight='balanced')
# Replace with: LogisticRegression(max_iter=300, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
```

---

### Phase 4: Add Rules 3 & 4 to unified_corpus.py (15 min)

1. **Add function** (see FIX_2_SYNC_RULES_ENFORCEMENT.md for full code)
2. **Replace lines 120-134** with call to `apply_sync_rules()`

Quick ref:
```python
# Add after deduplication, before shuffle:
unified_df = apply_sync_rules(unified_df, INPUT_DIR)
```

---

## 🔄 Phase 5-6: Setup & Regenerate (10 Minutes)

### Phase 5: Install Dependencies (5 min)

```bash
pip install -r requirements.txt --upgrade

# Verify
python -c "
import ftfy, conllu, requests, sklearn
print('✅ All dependencies installed!')
print(f'  sklearn: {sklearn.__version__}')
"
```

---

### Phase 6: Regenerate Balanced Corpus (5 min)

```bash
python data_preprocessing/unified_corpus.py

# Expected output shows:
# [RULE 3] Filtering token_count < 2...
# [RULE 4] Applying 3x class balance cap...
# Final imbalance ratio: 1.05x
# Unification complete! 248,600 rows.
```

---

## 🤖 Phase 7: Retrain Models (4-6 Hours - AUTOMATIC)

```bash
echo "Starting retraining... go get coffee ☕"
python src/advanced_training.py

# This will automatically:
# - Load balanced corpus
# - Tune hyperparameters
# - Train both models
# - Save advanced_lr.joblib and advanced_mnb.joblib
# - Generate reports
```

**During retraining, you can:**
- Read AUDIT_FIX_PROMPT.md for deep understanding
- Prepare Day 12 deployment checklist

---

## 📊 Phase 8: Generate Evaluation (15 Minutes)

### Step 1: Create Script

```bash
mkdir -p scripts

# Copy content from FIX_4_CONFUSION_MATRIX_EVALUATION.md
# into scripts/confusion_matrix_evaluation.py
```

### Step 2: Run Evaluation

```bash
python scripts/confusion_matrix_evaluation.py

# Expected: confusion_matrix_advanced_lr.csv, etc.
```

---

## ✅ Phase 9: Validation (30 Minutes)

### Test 1: sklearn Compatibility

```bash
python -c "
import sklearn
print(f'sklearn: {sklearn.__version__}')
assert sklearn.__version__[0:3] != '1.7', 'sklearn 1.7+ found!'
print('✅ OK')
"
```

### Test 2: Load Models

```bash
python -c "
import joblib
model = joblib.load('models/advanced_lr.joblib')
print('✅ Model loaded')
print(f'Prediction: {model.predict([\"hello world\"])}')
"
```

### Test 3: Corpus Balance

```bash
python -c "
import pandas as pd
df = pd.read_csv('preprocessed_dataset/unified_corpus.csv')
ratio = df['iso_code'].value_counts().max() / df['iso_code'].value_counts().min()
print(f'Imbalance: {ratio:.2f}x')
assert ratio < 1.1, 'Too imbalanced!'
print('✅ OK')
"
```

### Test 4: Chinese F1

```bash
grep "zh.*f1-score" reports/advanced_metrics.txt
# Should show F1 >= 0.90
```

### Test 5: Audit Report

```bash
python -c "
import json
with open('preprocessed_dataset/audit_report.json') as f:
    r = json.load(f)
print(f'Rows: {r[\"total_rows\"]:,}')
print(f'Imbalance: {r[\"imbalance_ratio\"]:.2f}x')
print('✅ OK')
"
```

### Test 6: Gradio App

```bash
timeout 30 python app.py || true
# Should launch without sklearn errors
```

---

## ✅ Success Checklist

```
✅ Phase 1: requirements.txt updated
✅ Phase 2: baseline_training.py patched
✅ Phase 3: advanced_training.py patched
✅ Phase 4: unified_corpus.py enhanced
✅ Phase 5: Dependencies installed
✅ Phase 6: Corpus regenerated (248.6k rows)
✅ Phase 7: Models retrained
✅ Phase 8: Confusion matrices generated
✅ Phase 9: All tests passed
```

---

## 🚀 Next Steps

```bash
# Commit changes
git add -A
git commit -m "Day 11: Apply all audit fixes - sklearn compat, corpus balance, evaluation"
git push origin main

# Deploy Day 12
# (your deployment process here)
```

---

## 🆘 Troubleshooting

| Error | Solution |
|-------|----------|
| `ValueError: multi_class must be in ('auto', 'multinomial')` | Run: `pip install 'scikit-learn<1.7.0'` |
| `No module named 'ftfy'` | Run: `pip install ftfy>=6.0.0` |
| `Model accuracy dropped` | Expected! Chinese F1 should improve +0.48. Check audit_report.json |
| `AttributeError: module has no attribute 'Blocks'` | Run: `pip install 'gradio<5.0.0'` |

---

**Status:** Ready to execute  
**Estimated completion:** 5-7 hours (mostly automatic)  
**Deployment target:** Day 12 🚀