"""
Language Detector — Gradio frontend.

Backend hook
------------
Replace the body of `_detect_language_backend` with your trained model call.
The function receives raw user text and must return the dict described in its
docstring.  Everything else in this file is UI/layout code.
"""

import os
import sys
import warnings
import pandas as pd
from datetime import datetime
from collections import Counter
import gradio as gr
import joblib

# Suppress sklearn version-mismatch warnings (models saved on 1.6.1, running 1.7.x)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Model loading ─────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

_MODELS: dict = {}

def _load_models() -> None:
    candidates = {
        "Advanced Logistic Regression": "models/advanced_lr.joblib",
        "Advanced Naive Bayes":         "models/advanced_mnb.joblib",
    }
    for name, rel in candidates.items():
        path = os.path.join(_BASE_DIR, rel)
        if os.path.exists(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _MODELS[name] = joblib.load(path)
            print(f"[model] Loaded: {name}")
        else:
            print(f"[model] Not found: {path}")

_load_models()

# Radio choices and default shown in the UI
_RADIO_CHOICES = list(_MODELS.keys()) if _MODELS else ["Demo Mode"]
_RADIO_DEFAULT = "Advanced Logistic Regression" if "Advanced Logistic Regression" in _MODELS else _RADIO_CHOICES[0]
_RADIO_INFO    = "LR: 97.2% acc  ·  NB: 96.4% acc" if _MODELS else "No trained models found — using rule-based demo"


# ── Language metadata ─────────────────────────────────────────────────────────

LANG_FLAGS = {
    "en": "🇬🇧", "ms": "🇲🇾", "id": "🇮🇩",
    "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷",
    "fr": "🇫🇷", "de": "🇩🇪", "es": "🇪🇸", "ar": "🇸🇦",
}

LANG_NAMES = {
    "en": "English", "ms": "Malay", "id": "Indonesian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "fr": "French",  "de": "German",  "es": "Spanish", "ar": "Arabic",
}

LANG_COLORS = {
    "en": "#3B82F6", "ms": "#10B981", "id": "#F59E0B",
    "zh": "#EF4444", "ja": "#EC4899", "ko": "#8B5CF6",
    "fr": "#6366F1", "de": "#14B8A6", "es": "#F97316", "ar": "#84CC16",
}


# ── Backend ───────────────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    try:
        from src.preprocess import preprocess_text
        result = preprocess_text(text)
        return result if result else text.lower().strip()
    except Exception:
        return text.lower().strip()


def _script_shortcut(text: str) -> list[tuple[str, float]] | None:
    """
    For scripts that map unambiguously to one language, skip the ML model.
    Returns a ranked list or None (meaning: let the ML model handle it).
    """
    has_hiragana  = any(0x3040 <= ord(c) <= 0x309F for c in text)
    has_katakana  = any(0x30A0 <= ord(c) <= 0x30FF for c in text)
    has_hangul    = any(0xAC00 <= ord(c) <= 0xD7AF or
                        0x1100 <= ord(c) <= 0x11FF for c in text)
    has_arabic    = any(0x0600 <= ord(c) <= 0x06FF for c in text)
    has_cjk       = any(0x4E00 <= ord(c) <= 0x9FFF for c in text)

    if has_hiragana or has_katakana:
        return [("ja", 0.98), ("zh", 0.01), ("ko", 0.01)]
    if has_hangul:
        return [("ko", 0.98), ("ja", 0.01), ("zh", 0.01)]
    if has_arabic:
        return [("ar", 0.98), ("ms", 0.01), ("en", 0.01)]
    if has_cjk:
        return [("zh", 0.97), ("ja", 0.02), ("ko", 0.01)]
    return None   # Latin / other — use ML model


# Lexical markers that strongly distinguish Malay from Indonesian
_MS_ONLY = {"kerana", "awak", "polis", "bas", "berbeza", "telefon", "doktor",
            "manakala", "walau", "sahaja", "sekadar", "ialah", "iaitu",
            "daripada", "kepada", "mempunyai", "hendak", "sudah", "sudahkah"}
_ID_ONLY = {"karena", "kamu", "enggak", "nggak", "gimana", "banget", "dong",
            "deh", "sih", "aja", "udah", "gak", "emang", "kayak", "polisi",
            "berbeda", "telepon", "dokter", "juga", "hanya", "bisa", "akan"}


def _lexical_boost(text: str, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    When the model is uncertain between Malay and Indonesian (top-2 are ms/id
    with < 0.85 confidence), apply lexical rules using language-exclusive words
    to nudge the result toward the correct language.
    """
    top_iso, top_conf = ranked[0]
    if top_iso not in ("ms", "id") or top_conf >= 0.85:
        return ranked

    words = set(text.lower().split())
    ms_hits = len(words & _MS_ONLY)
    id_hits = len(words & _ID_ONLY)

    if ms_hits == id_hits:
        return ranked  # no signal, leave unchanged

    winner  = "ms" if ms_hits > id_hits else "id"
    loser   = "id" if winner == "ms" else "ms"
    boost   = min(0.15 * abs(ms_hits - id_hits), 0.25)

    probs = dict(ranked)
    probs[winner] = min(probs.get(winner, 0) + boost, 0.99)
    probs[loser]  = max(probs.get(loser,  0) - boost, 0.01)

    return sorted(probs.items(), key=lambda t: t[1], reverse=True)


def _ml_predict(processed: str, model_name: str) -> list[tuple[str, float]]:
    """Run the sklearn pipeline and return ranked (iso, prob) pairs."""
    pipeline = _MODELS[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_probs = pipeline.predict_proba([processed])[0]
    classes = [str(c) for c in pipeline.classes_]
    ranked  = sorted(zip(classes, (float(p) for p in raw_probs)),
                     key=lambda t: t[1], reverse=True)
    ranked  = [(iso, p) for iso, p in ranked if iso != "unknown"]
    return _lexical_boost(processed, ranked)


def _demo_predict(text: str) -> list[tuple[str, float]]:
    """Rule-based fallback when no model is loaded."""
    words = set(text.lower().split())
    if words & {"berkenaan", "kepada", "kerana", "dengan", "untuk", "adalah", "boleh"}:
        return [("ms", 0.78), ("id", 0.18), ("en", 0.04)]
    if words & {"saya", "aku", "tidak", "yang", "bisa", "juga", "dari", "ini"}:
        return [("id", 0.55), ("ms", 0.40), ("en", 0.05)]
    if words & {"je", "les", "des", "une", "vous", "nous", "est", "dans"}:
        return [("fr", 0.88), ("es", 0.07), ("en", 0.05)]
    if words & {"ich", "die", "der", "das", "und", "ist", "nicht", "mit"}:
        return [("de", 0.87), ("en", 0.08), ("fr", 0.05)]
    if words & {"hola", "gracias", "por", "que", "esta", "para", "como", "pero"}:
        return [("es", 0.82), ("fr", 0.10), ("en", 0.08)]
    return [("en", 0.85), ("ms", 0.08), ("id", 0.07)]


def _detect_language_backend(text: str, model_name: str = "") -> dict:
    """
    Detection pipeline:
      1. Script shortcut  — handles Arabic / CJK / Hangul / Hiragana with ~98% accuracy
      2. ML model         — handles Latin-script languages (LR 97.1%, NB 93.9%)
      3. Demo fallback    — keyword rules when no model is loaded
    """
    # Step 1: script-based shortcut (unambiguous non-Latin scripts)
    ranked = _script_shortcut(text)

    # Step 2: ML model for everything else
    if ranked is None:
        processed = _preprocess(text)
        if model_name in _MODELS:
            ranked = _ml_predict(processed, model_name)
        else:
            ranked = _demo_predict(processed)

    top_iso, top_conf = ranked[0]
    return {
        "language":     LANG_NAMES.get(top_iso, top_iso.upper()),
        "iso_code":     top_iso,
        "confidence":   top_conf,
        "alternatives": [
            {"Language":    LANG_NAMES.get(iso, iso.upper()),
             "ISO Code":    iso,
             "Confidence":  f"{float(conf) * 100:.1f}%"}
            for iso, conf in ranked[:3]
        ],
    }


# ── Column names ──────────────────────────────────────────────────────────────

HISTORY_COLS = ["Text", "Detected Language", "Confidence", "Model", "Timestamp"]


# ── HTML builders ─────────────────────────────────────────────────────────────

def _lang_card(language: str = "", iso_code: str = "") -> str:
    if not language:
        return """
        <div class="result-card">
          <div class="result-label">🌐 Detected Language</div>
          <div class="result-empty">Run a detection to see results</div>
        </div>"""
    flag  = LANG_FLAGS.get(iso_code, "🌐")
    color = LANG_COLORS.get(iso_code, "#6366F1")
    return f"""
    <div class="result-card" style="border-left: 4px solid {color}">
      <div class="result-label">🌐 Detected Language</div>
      <div class="lang-row">
        <span class="lang-flag">{flag}</span>
        <span class="lang-name">{language}</span>
      </div>
      <span class="iso-pill" style="background:{color}22;color:{color};">{iso_code.upper()}</span>
    </div>"""


def _conf_card(confidence: float = -1) -> str:
    if confidence < 0:
        return """
        <div class="result-card">
          <div class="result-label">🎯 Confidence Score</div>
          <div class="result-empty">Run a detection to see results</div>
        </div>"""
    pct = confidence * 100
    if pct >= 80:
        color, label = "#10B981", "High confidence"
    elif pct >= 60:
        color, label = "#F59E0B", "Moderate confidence"
    elif pct >= 40:
        color, label = "#EF4444", "Low confidence"
    else:
        color, label = "#6B7280", "Ambiguous — try more text"
    return f"""
    <div class="result-card" style="border-left: 4px solid {color}">
      <div class="result-label">🎯 Confidence Score</div>
      <div class="conf-pct" style="color:{color}">{pct:.1f}%</div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{pct:.1f}%;background:{color}"></div>
      </div>
      <span class="conf-pill" style="background:{color}22;color:{color}">{label}</span>
    </div>"""


def _alts_html(alternatives: list[dict] | None = None) -> str:
    if not alternatives:
        return """
        <div class="alts-wrap">
          <div class="alts-empty">Top predictions will appear here after detection.</div>
        </div>"""
    medals = ["🥇", "🥈", "🥉"]
    rows = ""
    for i, alt in enumerate(alternatives[:3]):
        iso   = alt["ISO Code"]
        flag  = LANG_FLAGS.get(iso, "🌐")
        color = LANG_COLORS.get(iso, "#6366F1")
        pct   = float(alt["Confidence"].replace("%", ""))
        rows += f"""
        <div class="alt-row">
          <span class="alt-medal">{medals[i]}</span>
          <span class="alt-flag">{flag}</span>
          <div class="alt-info">
            <span class="alt-lang">{alt["Language"]}</span>
            <span class="alt-iso" style="background:{color}22;color:{color}">{iso.upper()}</span>
          </div>
          <div class="alt-bar-wrap">
            <div class="alt-bar-fill" style="width:{pct}%;background:{color}88"></div>
          </div>
          <span class="alt-pct" style="color:{color}">{alt["Confidence"]}</span>
        </div>"""
    return f'<div class="alts-wrap">{rows}</div>'


def _stat_card(icon: str, value, label: str, color: str, css_cls: str = "") -> str:
    return f"""
    <div class="stat-card {css_cls}">
      <div class="stat-icon">{icon}</div>
      <div class="stat-value" style="color:{color}">{value}</div>
      <div class="stat-label">{label}</div>
    </div>"""


def _empty_stat_cards() -> tuple[str, str, str]:
    return (
        _stat_card("📊", "0", "Total Detections", "#6366F1", "indigo"),
        _stat_card("🌍", "0", "Unique Languages",  "#10B981", "green"),
        _stat_card("🎯", "—", "Avg Confidence",    "#F59E0B", "amber"),
    )


# ── Data helpers ──────────────────────────────────────────────────────────────

def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLS)

def _empty_freq_df() -> pd.DataFrame:
    return pd.DataFrame({"Language": [], "Count": []})

def _history_to_df(history: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(history, columns=HISTORY_COLS) if history else _empty_history_df()

def _compute_stats(history: list[dict]) -> tuple[str, str, str, pd.DataFrame]:
    """Always returns exactly (total_html, unique_html, avg_html, freq_df)."""
    if not history:
        t_h, u_h, a_h = _empty_stat_cards()
        return t_h, u_h, a_h, _empty_freq_df()

    langs = [h["Detected Language"] for h in history]
    raws  = []
    for h in history:
        try:
            raws.append(float(h["Confidence"].replace("%", "")))
        except ValueError:
            pass
    avg   = f"{sum(raws)/len(raws):.1f}%" if raws else "—"

    freq     = Counter(langs)
    freq_df  = (
        pd.DataFrame({"Language": list(freq.keys()), "Count": list(freq.values())})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    t_h = _stat_card("📊", len(history),    "Total Detections", "#6366F1", "indigo")
    u_h = _stat_card("🌍", len(set(langs)), "Unique Languages",  "#10B981", "green")
    a_h = _stat_card("🎯", avg,             "Avg Confidence",    "#F59E0B", "amber")
    return t_h, u_h, a_h, freq_df


# ── Event handlers ────────────────────────────────────────────────────────────

def run_detection(text: str, history: list[dict], model_name: str):
    """All 9 outputs must always be returned in the same order."""
    t_h, u_h, a_h, freq_df = _compute_stats(history)

    if not text or not text.strip():
        return (
            _lang_card(), _conf_card(), _alts_html(),
            _history_to_df(history),
            t_h, u_h, a_h, freq_df,
            history,
        )

    result  = _detect_language_backend(text, model_name)
    iso     = result["iso_code"]
    lang_h  = _lang_card(result["language"], iso)
    conf_h  = _conf_card(result["confidence"])
    alts_h  = _alts_html(result["alternatives"])

    lang_disp = f"{result['language']} ({iso})"
    conf_disp = f"{result['confidence'] * 100:.1f}%"
    truncated = text[:40] + ("…" if len(text) > 40 else "")

    new_history = history + [{
        "Text":              truncated,
        "Detected Language": lang_disp,
        "Confidence":        conf_disp,
        "Model":             model_name,
        "Timestamp":         datetime.now().strftime("%H:%M:%S"),
    }]

    t_h, u_h, a_h, freq_df = _compute_stats(new_history)

    return (
        lang_h, conf_h, alts_h,
        _history_to_df(new_history),
        t_h, u_h, a_h, freq_df,
        new_history,
    )


def clear_detect():
    return "", _lang_card(), _conf_card(), _alts_html()


def clear_history_fn():
    t_h, u_h, a_h, freq_df = _compute_stats([])
    return _empty_history_df(), [], t_h, u_h, a_h, freq_df


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ─────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ── Hero — static gradient + dot-grid overlay ──────────────────── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #0f0c29 0%, #1e1b4b 40%, #2e1065 70%, #4c1d95 100%);
    border-radius: 24px;
    padding: 48px 56px 44px;
    margin-bottom: 32px;
    overflow: hidden;
}
/* dot-grid pattern */
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    color: rgba(255,255,255,0.92);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 999px;
    margin-bottom: 20px;
    backdrop-filter: blur(4px);
}
.hero h1 {
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    margin: 0 0 12px !important;
    letter-spacing: -1.5px !important;
    line-height: 1.08 !important;
}
.hero h1 .accent {
    background: linear-gradient(90deg, #a78bfa, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-size: 1.05rem !important;
    color: rgba(255,255,255,0.62) !important;
    margin: 0 0 24px !important;
    max-width: 500px !important;
    line-height: 1.6 !important;
}
.hero-langs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-lang-chip {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(255,255,255,0.78);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 5px 13px;
    border-radius: 999px;
    transition: background 0.2s, border-color 0.2s;
}
.hero-lang-chip:hover {
    background: rgba(255,255,255,0.14);
    border-color: rgba(255,255,255,0.28);
}

/* ── Detect button ──────────────────────────────────────────────── */
#detect-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    min-height: 50px !important;
}
#detect-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(99,102,241,0.65) !important;
}
#detect-btn:active {
    transform: translateY(0) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.40) !important;
}
#clear-btn {
    border-radius: 14px !important;
    font-weight: 600 !important;
    min-height: 50px !important;
    transition: opacity 0.15s !important;
}
#clear-btn:hover { opacity: 0.75 !important; }

/* ── Result cards ───────────────────────────────────────────────── */
.result-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-left: 4px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 30px 36px;
    min-height: 116px;
    margin-bottom: 14px;
}
.result-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
}
.result-empty {
    font-size: 0.88rem;
    color: var(--body-text-color-subdued);
    font-style: italic;
    margin-top: 6px;
    opacity: 0.7;
}
.lang-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.lang-flag { font-size: 1.9rem; line-height: 1; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); }
.lang-name { font-size: 1.65rem; font-weight: 800; color: var(--body-text-color); }
.iso-pill {
    display: inline-block;
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 6px;
}
.conf-pct {
    font-size: 2.4rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 12px;
}
.conf-track {
    background: var(--border-color-primary);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 12px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.15);
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.7s cubic-bezier(0.4,0,0.2,1);
    position: relative;
}
.conf-fill::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.30) 50%, transparent 100%);
    border-radius: inherit;
}
.conf-pill {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    letter-spacing: 0.04em;
}

/* ── Alternatives ───────────────────────────────────────────────── */
.alts-wrap {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    overflow: hidden;
    background: var(--background-fill-secondary);
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.alts-empty {
    padding: 32px;
    text-align: center;
    font-size: 0.9rem;
    font-style: italic;
    color: var(--body-text-color-subdued);
    opacity: 0.7;
}
.alt-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border-color-primary);
}
.alt-row:last-child { border-bottom: none; }
.alt-medal { font-size: 1.2rem; width: 26px; flex-shrink: 0; }
.alt-flag  { font-size: 1.4rem; flex-shrink: 0; filter: drop-shadow(0 1px 3px rgba(0,0,0,0.2)); }
.alt-info  { display: flex; align-items: center; gap: 8px; min-width: 150px; }
.alt-lang  { font-size: 0.95rem; font-weight: 600; color: var(--body-text-color); }
.alt-iso   {
    font-size: 0.6rem; font-weight: 800;
    letter-spacing: 0.14em; padding: 2px 8px;
    border-radius: 5px;
}
.alt-bar-wrap {
    flex: 1;
    background: var(--border-color-primary);
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.12);
}
.alt-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
    position: relative;
}
.alt-bar-fill::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25));
    border-radius: inherit;
}
.alt-pct { font-size: 0.88rem; font-weight: 700; min-width: 48px; text-align: right; }

/* ── Section label ──────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--body-text-color-subdued) !important;
    margin: 22px 0 10px !important;
}

/* ── Stat cards — per-card colored glow ─────────────────────────── */
.stat-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 20px;
    padding: 40px 28px;
    text-align: center;
}
.stat-icon  { font-size: 2.2rem; margin-bottom: 14px; }
.stat-value {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 8px;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--body-text-color-subdued);
}

/* ── History table ──────────────────────────────────────────────── */
#history-table table { font-size: 0.9rem !important; }
#history-table thead th {
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Buttons (secondary) ────────────────────────────────────────── */
#clear-history-btn {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: opacity 0.15s !important;
}
#clear-history-btn:hover { opacity: 0.75 !important; }
"""

# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["OK"],
    ["Saya suka makan nasi lemak dengan sambal yang pedas"],
    ["Aku suka makan bakso dan mie goreng enak sekali"],
    ["我喜欢学习人工智能和自然语言处理"],
    ["The quick brown fox jumps over the lazy dog"],
    ["Je suis étudiant en informatique à Paris"],
    ["Ich lerne Deutsch seit drei Jahren"],
    ["No me hagas daño, por favor"],
]

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Language Detector") as demo:

    gr.HTML("""
    <div class="hero">
      <div class="hero-content">
        <div class="hero-eyebrow">✦ NLP &nbsp;·&nbsp; Text Classification</div>
        <h1>Language <span class="accent">Detector</span></h1>
        <p>Instantly identify any language — with confidence scores, ISO codes, and ranked alternatives.</p>
        <div class="hero-langs">
          <span class="hero-lang-chip">🇬🇧 English</span>
          <span class="hero-lang-chip">🇲🇾 Malay</span>
          <span class="hero-lang-chip">🇮🇩 Indonesian</span>
          <span class="hero-lang-chip">🇨🇳 Chinese</span>
          <span class="hero-lang-chip">🇯🇵 Japanese</span>
          <span class="hero-lang-chip">🇰🇷 Korean</span>
          <span class="hero-lang-chip">🇫🇷 French</span>
          <span class="hero-lang-chip">🇩🇪 German</span>
          <span class="hero-lang-chip">🇪🇸 Spanish</span>
          <span class="hero-lang-chip">🇸🇦 Arabic</span>
        </div>
      </div>
    </div>
    """)

    history_state = gr.State([])

    with gr.Tabs():

        # ── Tab 1 · Detect ────────────────────────────────────────────────
        with gr.Tab("Detect"):
            with gr.Row():
                # Left column — input
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Enter text",
                        placeholder="Type or paste any text here…",
                        lines=5,
                    )
                    model_selector = gr.Radio(
                        choices=_RADIO_CHOICES,
                        value=_RADIO_DEFAULT,
                        label="Detection Model",
                    )
                    with gr.Row():
                        detect_btn = gr.Button(
                            "🔍  Detect Language",
                            variant="primary",
                            elem_id="detect-btn",
                            scale=2,
                        )
                        clear_btn = gr.Button(
                            "✕  Clear",
                            elem_id="clear-btn",
                            scale=1,
                        )

                # Right column — results
                with gr.Column(scale=2):
                    lang_html = gr.HTML(_lang_card())
                    conf_html = gr.HTML(_conf_card())

            gr.Markdown("#### Top 3 Predictions", elem_classes=["section-label"])
            alts_html = gr.HTML(_alts_html())

            gr.Markdown("#### Try These Examples", elem_classes=["section-label"])
            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input],
                label="",
            )

        # ── Tab 2 · History ───────────────────────────────────────────────
        with gr.Tab("History"):
            gr.Markdown(
                "Every detection is logged here automatically. "
                "Switch back to **Detect** and run a detection to populate this table.",
                elem_classes=["section-label"],
            )
            history_table = gr.Dataframe(
                value=_empty_history_df(),
                headers=HISTORY_COLS,
                interactive=False,
                label="Detection Log",
                elem_id="history-table",
                wrap=True,
            )
            with gr.Row():
                gr.HTML("<div></div>")  # spacer
                clear_history_btn = gr.Button(
                    "🗑  Clear History",
                    variant="secondary",
                    elem_id="clear-history-btn",
                    scale=0,
                )

        # ── Tab 3 · Statistics ────────────────────────────────────────────
        with gr.Tab("Statistics"):
            gr.Markdown(
                "Live stats — refreshed automatically after every detection.",
                elem_classes=["section-label"],
            )
            with gr.Row():
                total_html  = gr.HTML(_stat_card("📊", "0", "Total Detections", "#6366F1", "indigo"))
                unique_html = gr.HTML(_stat_card("🌍", "0", "Unique Languages",  "#10B981", "green"))
                avg_html    = gr.HTML(_stat_card("🎯", "—", "Avg Confidence",    "#F59E0B", "amber"))

            gr.Markdown("#### Language Frequency", elem_classes=["section-label"])
            lang_freq_plot = gr.BarPlot(
                value=_empty_freq_df(),
                x="Language",
                y="Count",
                title="",
                x_title="Language",
                y_title="Detections",
                color="Language",
            )

    # ── Event wiring ──────────────────────────────────────────────────────────
    #   run_detection returns 9 values → 9 outputs (order must match exactly)

    all_detect_outputs = [
        lang_html, conf_html, alts_html,        # Tab 1 (3)
        history_table,                          # Tab 2 (1)
        total_html, unique_html, avg_html,      # Tab 3 (3)
        lang_freq_plot,                         # Tab 3 (1)
        history_state,                          #       (1)  total = 9
    ]

    detect_btn.click(
        fn=run_detection,
        inputs=[text_input, history_state, model_selector],
        outputs=all_detect_outputs,
    )

    clear_btn.click(
        fn=clear_detect,
        outputs=[text_input, lang_html, conf_html, alts_html],
    )

    clear_history_btn.click(
        fn=clear_history_fn,
        outputs=[history_table, history_state,
                 total_html, unique_html, avg_html, lang_freq_plot],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
