# FIX_3: REQUIREMENTS.TXT - VERSION PINNING

## The Problem

**File:** `requirements.txt`

**Current Issues:**
```
scikit-learn>=1.3.0          # ❌ Allows 1.7.0+ with breaking changes
gradio>=4.0.0                # ❌ Allows 5.0+ which might break UI
                             # ❌ MISSING: ftfy (D2 preprocessing)
                             # ❌ MISSING: conllu (D6 preprocessing)
                             # ❌ MISSING: requests (D6 downloads)
```

---

## The Solution

**Replace entire `requirements.txt` with:**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0,<1.7.0
joblib>=1.3.0
gradio>=4.0.0,<5.0.0
ftfy>=6.0.0
conllu>=4.0.0
requests>=2.25.0
```

---

## Quick Update

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
```

---

## Why Each Constraint

| Package | Constraint | Reason |
|---------|-----------|--------|
| scikit-learn | `<1.7.0` | multi_class='ovr' removed in 1.7 |
| gradio | `<5.0.0` | Major version breaking changes |
| ftfy | `>=6.0.0` | D2 mojibake recovery (currently missing!) |
| conllu | `>=4.0.0` | D6 UD treebank parsing (currently missing!) |
| requests | `>=2.25.0` | D6 file downloads (currently missing!) |

---

## Verify

```bash
pip install -r requirements.txt --upgrade

python -c "
import ftfy, conllu, requests, sklearn
print(f'✅ All dependencies OK')
print(f'  sklearn: {sklearn.__version__}')
print(f'  ftfy: {ftfy.__version__}')
"
```

---

## Impact

- ✅ Fresh install works end-to-end
- ✅ Protected from future breaking changes
- ✅ All preprocessing dependencies included
- ✅ sklearn compatible with models

---

## Time: 5 minutes