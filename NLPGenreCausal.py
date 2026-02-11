import os
from pathlib import Path
import calendar
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================================
# ✅ STREAMLIT CONFIG (MUST BE FIRST STREAMLIT CALL)
# ============================================================
st.set_page_config(layout="wide")
st.title("🎬 TV Genre Evolution Dashboard")


# ============================================================
# 0) LOAD DATA FIRST (combined_df)
# ============================================================
DATA_PATH = "Merged_Df_Cleaned.csv"  # change to .csv if needed

@st.cache_data(show_spinner=True)
def load_combined_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. If on Streamlit Cloud, make sure the file is in the repo "
            f"(or load via st.file_uploader / URL / S3)."
        )

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type. Use .parquet or .csv")

    expected = {"Show", "Genre", "Air Date", "Viewership (millions)"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"combined_df is missing required columns: {sorted(missing)}")

    return df

try:
    combined_df = load_combined_df(DATA_PATH)
except Exception as e:
    st.error(f"Data failed to load: {e}")
    st.stop()


# ============================================================
# APP INTRO
# ============================================================
st.markdown("""
*Emotional arcs, cultural shifts, and audience engagement at scale.*

---

## **The Problem**
Television genres are not static—they evolve alongside technology, culture, and shifting audience expectations.  
From post-9/11 security-infused dramas to streaming-era prestige series and meme-driven comedies, emotional arcs in scripted TV reflect and shape the **cultural zeitgeist**.  

Our analysis of **25,000+ episodes** spanning multiple decades shows that each genre develops its own emotional “grammar” over time, including patterns of pacing, tone, and act-level shifts that audiences subconsciously expect. These arcs often align with broader societal moments:  
- **Fear & patriotism** in early-2000s thrillers.  
- **Irony & sarcasm** in late-90s comedies.  
- **Surprise-driven suspense** in modern fantasy franchises.  

Yet without visibility into how these emotional structures evolve, creatives risk greenlighting concepts or shaping campaigns that miss the cultural moment, and fail to sustain viewership.

---

## **The Solution**
This dashboard translates **NLP, time-series modeling, and cultural analysis** into actionable creative intelligence.  

It reveals:  
- 🎭 **Genre-specific emotional signatures**  
- ⏳ **Act-level pacing patterns** that predict retention  
- 📆 **Cultural breakpoints** when genres pivot in tone  
- 🎯 **Proven emotional hooks** that historically boost next-episode viewership per genre  

---

## **Data Sources**
- 25,000+ TV scripts from closed-caption text sources.
- Nielsen ratings cited from Wikipedia and other published sources.
- Metadata from IMDb (season, year, episode, etc.).
""")


# ============================================================
# SHOW + SCOPE CONTROLS
# ============================================================
selected_show = st.selectbox("Select a TV Show", sorted(combined_df["Show"].dropna().unique()))
scope = st.radio("Scope", ["All data", "Selected show"], horizontal=True)

def get_scope_df(df: pd.DataFrame) -> pd.DataFrame:
    if scope == "Selected show":
        sdf = df[df["Show"] == selected_show].copy()
        if sdf.empty:
            st.warning("No rows for this show. Switching to All data.")
            return df
        return sdf
    return df


# ============================================================
# APPENDIX 1: NLP DICTIONARY
# ============================================================
st.markdown("## Appendix 1. NLP Dictionary")
st.caption("Look up any term by category or search. Terms mirror coding used in the dashboard.")

GLOSSARY = {
    "Story Act Designations": {
        "Act suffixes (_1, _2, _3)":
            "Acts 1, 2, and 3 within a TV episode, determined by script length to study how emotions vary across structure."
    },
    "Emotion & Psychological Engagement": {
        "Sd_Scaled":
            "[Mean/Scaled] Scaled standard deviation of overall emotion scores across acts. "
            "Derived from the sum of per-act SDs and divided by the mean to compare variability and intensity; "
            "conceptually linked to transportation (speed/depth of immersion).",
        "Anger": "[Percent] Percentage of words in an act expressing anger.",
        "Surprise": "[Percent] Percentage of words in an act expressing surprise.",
        "Disgust": "[Percent] Percentage of words in an act expressing disgust.",
        "Sadness": "[Percent] Percentage of words in an act expressing sadness.",
        "Neutral": "[Percent] Percentage of words in an act expressing neutrality.",
        "Fear": "[Percent] Percentage of words in an act expressing fear.",
        "Joy": "[Percent] Percentage of words in an act expressing joy.",
        "Positive": "[Percent] Percentage of words in an act expressing positive sentiment.",
        "Engaged": "[Mean] Composite score of psychological involvement or emotional investment."
    },
    "Linguistic & Narrative Complexity": {
        "Word Count": "[Total] Total number of words in an act.",
        "Analytic": "[Mean] Average score of analytical/formal/logical language.",
        "Clout": "[Mean] Average score of confidence/leadership/status-signaling language.",
        "Authenticity": "[Mean] Average score of honest, unfiltered, or self-revealing language.",
        "Tone": "[Mean] Average measure of positivity (scores < 50 indicate negativity).",
        "WPS (Words per Sentence)": "[Mean] Average number of words per sentence.",
        "Six Letter Word": "[Percent] Percentage of words with more than six letters.",
        "Dic (Dictionary)": "[Percent] Percentage of words captured in the reference dictionary."
    },
    "Cognitive Processing": {
        "Cogprocess": "[Mean] Words signaling active information processing and causation.",
        "Insight": "[Mean] Words indicating realizations or understanding.",
        "Cause": "[Mean] Words signaling causal relations.",
        "Discrep": "[Mean] Counterfactual/contrastive words (e.g., should, could).",
        "Tentative": "[Mean] Uncertainty/possibility words (e.g., maybe, perhaps).",
        "Certain": "[Mean] Absolute words (e.g., always, never).",
        "Differ": "[Mean] Differentiation/contrast words (e.g., but, else)."
    },
    "Perceptual Processing": {
        "Perceptual": "[Mean] Perception words (e.g., look, heard, feel).",
        "See": "[Mean] Visual perception words.",
        "Hear": "[Mean] Auditory perception words.",
        "Feel": "[Mean] Tactile/embodied sensation words."
    },
    "Motivational Drives": {
        "Drives": "[Mean] Motivational expressions.",
        "Affiliation": "[Mean] Social-relationship words (e.g., ally, friend).",
        "Achieve": "[Mean] Success/accomplishment words.",
        "Power": "[Mean] Dominance/hierarchy words.",
        "Reward": "[Mean] Benefit/prize words.",
        "Risk": "[Mean] Danger/uncertainty words."
    },
    "Spatial & Temporal": {
        "Relativ (Relativity)": "[Mean] Spatial relation words (e.g., area, bend, exit).",
        "Motion": "[Mean] Movement words (e.g., arrive, go, car).",
        "Space": "[Mean] Spatial direction/location words (e.g., down, in).",
        "Time": "[Mean] Temporal words (e.g., end, until, season)."
    },
}

flat_terms = [{"category": cat, "term": term, "definition": desc}
              for cat, terms in GLOSSARY.items()
              for term, desc in terms.items()]

q_search = st.text_input("Search terms (e.g., 'anger', 'WPS', 'transportation')").strip().lower()

if q_search:
    matches = [t for t in flat_terms if q_search in t["term"].lower() or q_search in t["definition"].lower()]
    if matches:
        options = [f"{m['term']}  —  {m['category']}" for m in matches]
        sel = st.selectbox("Matching terms", options)
        picked = matches[options.index(sel)]
        st.markdown(f"### {picked['term']}")
        st.caption(picked["category"])
        st.markdown(picked["definition"])
    else:
        st.info("No matches found. Try a different keyword.")
else:
    cat = st.selectbox("Category", list(GLOSSARY.keys()))
    term = st.selectbox("Term", list(GLOSSARY[cat].keys()))
    st.markdown(f"### {term}")
    st.caption(cat)
    st.markdown(GLOSSARY[cat][term])

with st.expander("Show full dictionary"):
    for cat, terms in GLOSSARY.items():
        st.markdown(f"**{cat}**")
        for term, desc in terms.items():
            st.markdown(f"- **{term}** — {desc}")


# ============================================================
# SECTION 0: DESCRIPTIVE STATS
# ============================================================
st.header("🧮 Descriptive Statistics for Emotion & Linguistic Features")
data = get_scope_df(combined_df)

base_prefixes = [
    'Anger','Surprise','Disgust','Sadness','Neutral','Fear','Joy',
    'Positive','Negative','Engaged','Not engaged',
    'WC','Analytic','Clout','Authentic','Tone','WPS',
    'Sixltr','Six_Let',
    'Dic','Dictionary',
    'Cogproc','Insight','Cause','Discrep','Tentat','Certain','Differ',
    'Percept','See','Hear','Feel','Drives','Affiliation','Achieve',
    'Power','Reward','Risk','Relativ','Motion','Space','Time',
    'Transportation'
]

synonym_map = {'Six_Let': 'Sixltr', 'Dictionary': 'Dic'}

def normalize_prefix(p: str) -> str:
    return synonym_map.get(p, p)

# ✅ IMPORTANT: do NOT build candidate_cols from string-casted names;
# use actual columns to avoid KeyError with non-string/MultiIndex cols
candidate_cols = []
for col in data.columns:
    colname = str(col)
    if any(colname.startswith(p) for p in base_prefixes):
        candidate_cols.append(col)

# de-duplicate + stable sort
candidate_cols = list(dict.fromkeys(candidate_cols))
candidate_cols = sorted(candidate_cols, key=lambda x: (str(x).split('_')[0], str(x)))

if not candidate_cols:
    st.info("No emotion/linguistic columns detected by prefix. Showing all columns so you can verify naming.")
    st.write(list(map(str, data.columns)))
else:
    st.subheader("Overall (All Acts Combined)")
    st.caption("Includes any column whose name starts with an emotion/linguistic prefix (e.g., `Anger_1`, `WC_2`, `Dic_3`).")
    st.dataframe(data[candidate_cols].describe().transpose(), use_container_width=True)

    for act in ('_1', '_2', '_3'):
        act_cols = [c for c in candidate_cols if str(c).endswith(act)]
        if act_cols:
            st.subheader(f"Act-level Summary: Act {act[-1]}")
            st.dataframe(data[act_cols].describe().transpose(), use_container_width=True)

    family = defaultdict(list)
    for c in candidate_cols:
        base = str(c).split('_')[0]
        base = normalize_prefix(base)
        family[base].append(c)

    collapsed = {}
    for base, cols in family.items():
        valid = [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]
        if valid:
            collapsed[base] = data[valid].mean(axis=1)

    if collapsed:
        collapsed_df = pd.DataFrame(collapsed)
        st.subheader("Collapsed Feature Families (Mean Across Acts)")
        st.dataframe(collapsed_df.describe().transpose(), use_container_width=True)


# ============================================================
# 🧪 CAUSALITY (SAFE / OPTIONAL)
# ============================================================
st.markdown("---")
st.subheader("🧪 Causality (Double ML & Causal Forest) — Continuous Only")
st.caption("Estimate treatment effects of a selected, actionable NLP feature on next-episode viewership.")

# ✅ Define TARGET_COL once (prevents NameError)
TARGET_COL = "Viewership (millions)"  # change if your season_df uses a different target column name

econml_ok = True
try:
    from econml.dml import LinearDML, CausalForestDML
except Exception as e:
    econml_ok = False
    st.info(
        "Causality (econml) is unavailable in this deployment.\n\n"
        "To enable it on Streamlit Cloud, use:\n"
        "- `runtime.txt` → `python-3.11`\n"
        "- `requirements.txt` → e.g. `econml==0.15.1` and `shap==0.43.0` (compatible pair)\n"
        f"(Import error: {e})"
    )

# Safe defaults to prevent NameError in narrative section below
t_feat = "(select a treatment feature above)"
X_cols = []
ate = lb = ub = None
cate = coverage = policy_value = None
q_policy = 20  # default targeting percent
mu = sd = np.nan
t_raw = np.array([])
aligned = pd.DataFrame()

if "season_df" not in st.session_state:
    st.info("No season-level data available yet.")
else:
    season_df = st.session_state["season_df"].copy()

    dupes = season_df.columns[season_df.columns.duplicated()].unique()
    if len(dupes) > 0:
        st.warning(f"Duplicate columns removed: {list(map(str, dupes))}")
        season_df = season_df.loc[:, ~season_df.columns.duplicated()].copy()

    if TARGET_COL not in season_df.columns:
        st.info(f"Target column '{TARGET_COL}' is missing in the season data.")
    else:
        nlp_prefixes = [
            "Anger","Surprise","Disgust","Sadness","Fear","Joy","Positive","Negative",
            "Analytic","Clout","Authentic","Tone","WPS","Six","Dic",
            "Cog","Insight","Cause","Discrep","Tentat","Certain","Differ",
            "Percep","See","Hear","Feel","Drives","Affiliation","Achieve",
            "Power","Reward","Risk","Relativ","Motion","Space","Time"
        ]

        nlp_candidates = [
            c for c in season_df.columns
            if (isinstance(c, tuple) and any(str(c[0]).startswith(p) for p in nlp_prefixes))
            or (isinstance(c, str) and any(str(c).startswith(p) for p in nlp_prefixes))
        ]

        with st.expander("Define treatment feature (continuous z-score)", expanded=True):
            if isinstance(season_df.columns, pd.MultiIndex):
                top_level_names = sorted({c[0] for c in season_df.columns})
                choices = [c for c in top_level_names if any(str(c).startswith(p) for p in nlp_prefixes)]
                t_feat = st.selectbox("Choose treatment (top level)", choices) if choices else "(no NLP found)"
            else:
                t_feat = st.selectbox("Choose treatment", nlp_candidates or ["(no NLP found)"])

        with st.expander("Controls (X) and model settings", expanded=True):
            default_ctrl = ["Season", "Episode", "WC", "WPS", "Analytic", "Clout", "Tone", "Year", "Time_in_minutes"]

            def _is_treatment_col(col):
                if isinstance(col, tuple):
                    return col[0] == t_feat
                return col == t_feat

            ctrl_pool = [c for c in default_ctrl if c in season_df.columns]
            extra = [c for c in season_df.columns if (c not in ctrl_pool) and not _is_treatment_col(c)]
            ctrl_candidates = [c for c in (ctrl_pool + extra) if isinstance(c, str)]

            X_cols = st.multiselect("Select controls (keep parsimonious)", ctrl_candidates, default=ctrl_candidates[:6])

            do_run_dml = st.checkbox("Run Double ML (LinearDML)", value=True, disabled=not econml_ok)
            do_run_cf = st.checkbox("Run Causal Forest (CausalForestDML)", value=True, disabled=not econml_ok)

        if not econml_ok:
            st.warning("Install/enable econml to run this section.")
        else:
            def pick_treatment_series(df, t_feat_name):
                if isinstance(df.columns, pd.MultiIndex):
                    subs = [col for col in df.columns if col[0] == t_feat_name]
                    if not subs:
                        st.error(f"No subcolumns found under treatment '{t_feat_name}'.")
                        return None
                    labels = [" / ".join(map(str, c)) for c in subs]
                    chosen_label = st.selectbox("Select treatment subcolumn", labels, key="t_sub")
                    col = subs[labels.index(chosen_label)]
                    st.caption(f"Using treatment column: {col}")
                    return pd.to_numeric(df[col], errors="coerce")
                else:
                    st.caption(f"Using treatment column: {t_feat_name}")
                    return pd.to_numeric(df[t_feat_name], errors="coerce")

            ready = False
            if t_feat and X_cols:
                if not isinstance(season_df.columns, pd.MultiIndex):
                    cols_for_work = [TARGET_COL, t_feat] + X_cols
                    work = season_df[cols_for_work].copy()
                else:
                    work = season_df[[TARGET_COL] + X_cols].copy()

                work = work.apply(pd.to_numeric, errors="coerce", axis=0)

                if isinstance(work.columns, pd.MultiIndex):
                    Y = pd.to_numeric(season_df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
                else:
                    Y = work[TARGET_COL].to_numpy(dtype=float)

                t_series = pick_treatment_series(season_df, t_feat)

                if t_series is not None:
                    t_series = t_series.loc[season_df.index]
                    base = pd.DataFrame({TARGET_COL: Y}, index=season_df.index)
                    base["T_raw"] = t_series

                    X_df = season_df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                    aligned = pd.concat([base, X_df], axis=1).dropna(subset=[TARGET_COL, "T_raw"])

                    Y = aligned[TARGET_COL].to_numpy(dtype=float)
                    t_raw = aligned["T_raw"].to_numpy(dtype=float)
                    X = aligned[X_cols].to_numpy(dtype=float)

                    mu = float(np.nanmean(t_raw))
                    sd = float(np.nanstd(t_raw))
                    T = (t_raw - mu) / (sd + 1e-9)

                    if len(Y) == len(T) == X.shape[0] and X.shape[0] >= 30 and X.shape[1] >= 2:
                        ready = True
                    else:
                        st.info(f"Not enough aligned data (rows={X.shape[0]}, features={X.shape[1]}). Need ≥30 rows and ≥2 features.")
                else:
                    st.info("Pick a valid treatment column.")
            else:
                st.info("Select a treatment and at least one control to enable modeling.")

            if ready and do_run_dml:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import LassoCV

                    if Y.ndim == 2 and Y.shape[1] == 1:
                        Y = Y.ravel()
                    if T.ndim == 2 and T.shape[1] == 1:
                        T = T.ravel()

                    if T.ndim > 1:
                        st.error(f"T has shape {T.shape} (multi-output). Pick a single subcolumn for '{t_feat}'.")
                    else:
                        st.caption(f"Shapes — Y: {Y.shape}, T: {T.shape}, X: {X.shape}")

                        model_y = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1)
                        model_t = LassoCV(cv=3, random_state=42)

                        dml = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, cv=3, random_state=42)
                        dml.fit(Y=Y, T=T, X=X)

                        ate = float(dml.ate(X=X))
                        lb, ub = dml.ate_interval(X=X)

                        st.markdown("**Double ML (LinearDML) — ATE (continuous)**")
                        st.write(f"Estimated ATE (per +1 SD in {t_feat}): **{ate:.4f}**")
                        st.caption(f"95% CI: [{lb:.4f}, {ub:.4f}]")

                        st.markdown("**Policy Rule (Continuous): Aim for +k SD increase**")
                        k = st.slider("Target increase (k SD)", 0.1, 1.0, 0.5, 0.1, key="k_sd_dml")
                        target_raw = mu + k * sd
                        gap_raw = np.maximum(0.0, target_raw - t_raw)
                        gap_sd = np.minimum(k, gap_raw / (sd + 1e-9))
                        expected_lift = ate * gap_sd

                        rule_df = pd.DataFrame({
                            "Row": aligned.index,
                            f"{t_feat}_raw": t_raw,
                            f"{t_feat}_z": (t_raw - mu) / (sd + 1e-9),
                            "target_raw": target_raw,
                            "raise_by_raw": gap_raw,
                            "raise_by_SD": gap_sd,
                            "expected_lift": expected_lift
                        }).sort_values("expected_lift", ascending=False)

                        st.dataframe(rule_df, use_container_width=True)
                except Exception as e:
                    st.error(f"LinearDML failed: {e}")

            if ready and do_run_cf:
                try:
                    from sklearn.ensemble import RandomForestRegressor

                    mask_all = np.isfinite(Y) & np.isfinite(T) & np.isfinite(X).all(axis=1)
                    Y_cf, T_cf, X_cf = Y[mask_all], T[mask_all], X[mask_all]

                    if not (len(Y_cf) == len(T_cf) == X_cf.shape[0]):
                        st.error("Causal Forest aborted: arrays still not aligned.")
                    else:
                        cf = CausalForestDML(
                            model_t=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1),
                            model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1),
                            n_estimators=1500, min_samples_leaf=10, random_state=42
                        )
                        cf.fit(Y=Y_cf, T=T_cf, X=X_cf)
                        cate = cf.effect(X_cf)

                        st.markdown("**Causal Forest — CATE summary**")
                        st.write(f"Mean CATE: {np.nanmean(cate):.4f} | Median: {np.nanmedian(cate):.4f}")

                        q_policy = st.slider("Target top q% episodes by predicted CATE", 5, 50, 20, step=5, key="q_cf")
                        thr = np.nanpercentile(cate, 100 - q_policy)
                        take = cate >= thr
                        coverage = float(np.mean(take))
                        policy_value = float(np.nanmean(cate[take])) if np.any(take) else float("nan")

                        c1, c2 = st.columns(2)
                        c1.metric("Coverage", f"{coverage*100:.1f}%")
                        c2.metric("Avg predicted lift (targeted)", f"{policy_value:.4f}")
                except Exception as e:
                    st.error(f"Causal Forest failed: {e}")


# ============================================================
# 🔎 AUTOMATED NARRATIVE SUMMARY (SAFE DEFAULTS FIRST)
# ============================================================
def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def interpret_ate(ate_val, lb_val, ub_val, t_feat_name, target_name):
    if ate_val is None or lb_val is None or ub_val is None:
        return "Run Double ML to estimate the overall effect (ATE)."
    if lb_val > 0:
        return (f"Increasing **{t_feat_name}** likely **increases** **{target_name}** on average "
                f"(ATE {ate_val:.4f} per +1 SD; 95% CI [{lb_val:.4f}, {ub_val:.4f}]).")
    if ub_val < 0:
        return (f"Increasing **{t_feat_name}** likely **decreases** **{target_name}** on average "
                f"(ATE {ate_val:.4f} per +1 SD; 95% CI [{lb_val:.4f}, {ub_val:.4f}]).")
    return (f"The **overall effect** of **{t_feat_name}** on **{target_name}** is **inconclusive** "
            f"(ATE {ate_val:.4f} per +1 SD; 95% CI [{lb_val:.4f}, {ub_val:.4f}] spans 0).")

controls_str = ", ".join([c for c in X_cols if c != t_feat]) if X_cols else "(none)"
ate_available = (ate is not None) and (lb is not None) and (ub is not None)
cf_available = (cate is not None) and (coverage is not None) and (policy_value is not None)

st.markdown("### 📗 What the terms mean (plain English)")
st.markdown(f"""
- **Treatment (T)** — the knob we imagine turning: **{t_feat}** (z-score; “+1 SD” = one standard deviation above the mean).
- **Outcome (Y)** — what we want to move: **{TARGET_COL}**.
- **Controls (X)** — context we adjust for: **{controls_str}**.
- **ATE** — average change in **{TARGET_COL}** for **+1 SD** in **{t_feat}** across all episodes.
- **CATE** — episode-specific effect given its context X.
- **Coverage** — share of episodes targeted by the rule (top **{int(q_policy)}%** by predicted CATE).
""")

st.markdown("### 🧾 What we ran")
st.markdown(
    f"We treated **{t_feat}** as **continuous (z-score)** and adjusted for **{controls_str}**. "
    "We estimated overall effects with **Double ML** and heterogeneity with a **Causal Forest**."
)

st.markdown("### 🌍 Overall effect (ATE)")
if ate_available:
    st.markdown(interpret_ate(ate, lb, ub, t_feat, TARGET_COL))
else:
    st.markdown("Run Double ML to estimate the **overall effect (ATE)**.")

st.markdown("### 🎯 Who to target (CATE-based policy)")
if cf_available:
    st.markdown(
        f"Target the **top {int(q_policy)}%** by predicted CATE. "
        f"**Coverage:** {_fmt(coverage*100, 1)}%. "
        f"**Predicted gain (targeted, per +1 SD):** {_fmt(policy_value)}."
    )
else:
    st.markdown("Run **Causal Forest** to enable targeting, coverage, and predicted gains.")


# ============================================================
# 📉 TIME SERIES DECOMPOSITION
# ============================================================
st.header("📉 Time Series Decomposition")
st.markdown("""
**How to read these components**
- **Trend:** Long-term direction of viewership after smoothing out seasonality and noise.
- **Seasonality:** Recurring within-year patterns (e.g., fall premieres, holidays).
- **Residuals:** Irregular, unexplained variation not captured by trend/seasonality.
- **Original Data:** Observed monthly totals for the genre group.
""")

def _classify(value, thresholds, labels):
    for t, lab in zip(thresholds, labels):
        if value <= t:
            return lab
    return labels[-1]

def _month_abbr_list(month_indices):
    return [calendar.month_abbr[int(m)] for m in month_indices]

def summarize_and_recommend(group_name, trend_s, seas_s, resid_s, orig_series):
    tr = trend_s.dropna()
    if len(tr) >= 3:
        x_years = (tr.index.view("int64") / 1e9 / 86400 / 365.25)
        x_years = x_years - x_years.min()
        y = tr.values
        slope, _ = np.polyfit(x_years, y, 1)
    else:
        slope = np.nan

    if np.isnan(slope):
        trend_dir, slope_emoji = "unclear", "❔"
    elif slope > 0.02:
        trend_dir, slope_emoji = "rising", "📈"
    elif slope < -0.02:
        trend_dir, slope_emoji = "falling", "📉"
    else:
        trend_dir, slope_emoji = "stable", "➖"

    trend_vol_label = _classify(tr.std(), [0.25, 0.75, 1.5], ["very low", "low", "moderate", "high", "very high"])

    ss = seas_s.dropna()
    seas_amp = (ss.quantile(0.95) - ss.quantile(0.05))
    seas_strength_ratio = 0 if orig_series.std() == 0 else seas_amp / max(orig_series.std(), 1e-9)
    seas_strength_label = _classify(seas_strength_ratio, [0.1, 0.25, 0.5], ["weak", "mild", "moderate", "strong"])

    if len(ss) > 0:
        month_means = ss.groupby(ss.index.month).mean().sort_values(ascending=False)
        top_months = _month_abbr_list(list(month_means.index[:2]))
    else:
        top_months = []

    resid_vol_label = _classify(resid_s.dropna().std(), [0.15, 0.35, 0.75], ["very low", "low", "moderate", "high", "very high"])

    aligned_corr = np.nan
    if len(tr) > 0 and len(ss) > 0:
        corr_df = pd.concat([tr, ss], axis=1, join="inner").dropna()
        if not corr_df.empty:
            aligned_corr = corr_df.iloc[:, 0].corr(corr_df.iloc[:, 1])

    if np.isnan(aligned_corr):
        corr_label = "unclear"
    elif aligned_corr >= 0.4:
        corr_label = "aligned"
    elif aligned_corr <= -0.4:
        corr_label = "counter-moving"
    else:
        corr_label = "weakly related"

    when = " & ".join(top_months) if top_months else "no consistent peak months"
    exec_oneliner = f"**TL;DR:** {group_name} is **{trend_dir}**; plan key beats around **{when}**; expect **{resid_vol_label}** unpredictability."

    actions = []
    if trend_dir == "rising":
        actions.append("Scale winning formats/subgenres; increase supply and media weight in peak months.")
    elif trend_dir == "falling":
        actions.append("Audit pacing/emotional arcs vs. successful genre norms; refresh promos to foreground proven hooks.")
    else:
        actions.append("Maintain cadence while A/B testing creative and release timing for lift.")

    if seas_strength_label in ["moderate", "strong"] and top_months:
        actions.append(f"Time premieres/tentpoles in **{when}**; concentrate paid media in those windows.")
    else:
        actions.append("Use an always-on strategy; spread launches and budget to avoid over-reliance on specific windows.")

    if resid_vol_label in ["high", "very high"]:
        actions.append("Investigate spikes/dips (events, platform changes, controversies); widen forecast bands; stress-test scenarios.")
    else:
        actions.append("Standard forecast bands suffice; monitor anomalies but prioritize trend/seasonality signals.")

    if corr_label == "aligned":
        actions.append("Bundle big narrative beats with seasonal peaks to amplify impact.")
    elif corr_label == "counter-moving":
        actions.append("Consider counter-programming outside typical peaks to avoid congestion.")

    return exec_oneliner, actions

data_ts = get_scope_df(combined_df).copy()
data_ts["Air Date"] = pd.to_datetime(data_ts["Air Date"], errors="coerce")
data_ts = data_ts.dropna(subset=["Air Date"])
today = pd.Timestamp.today().normalize()
data_ts = data_ts[(data_ts["Air Date"] >= pd.Timestamp("1985-01-01")) & (data_ts["Air Date"] <= today)]
data_ts = data_ts.set_index("Air Date").sort_index()

genre_groups = {
    "Comedy & Satire": ["Sitcom", "Comedy-Drama", "Animated sitcom", "Comedy", "Comedy horror"],
    "Drama": ["Drama", "Medical drama", "Family drama", "Period drama", "Political drama", "Psychological drama", "Serial drama"],
    "Crime, Law & Justice": ["Crime drama", "Police procedural", "Legal drama", "Legal thriller", "Crime", "Psychological thriller"],
    "Action & Adventure": ["Action", "Action-adventure", "Superhero", "Adventure", "Action fiction"],
    "Fantasy & Sci-Fi": ["Fantasy", "Science fiction", "Science fantasy", "Supernatural", "Supernatural drama"],
    "Horror & Suspense": ["Horror", "Mystery", "Psychological horror", "Thriller"]
}

for group_name, subgenres in genre_groups.items():
    group_df = data_ts[data_ts["Genre"].isin(subgenres)]
    if group_df.empty:
        continue

    group_ts = group_df[["Viewership (millions)"]].resample("M").sum()
    group_ts = group_ts[group_ts["Viewership (millions)"] > 0].ffill(limit=2)
    group_ts = group_ts[group_ts.index <= today]

    if group_ts.shape[0] < 24:
        continue

    decomposition = seasonal_decompose(group_ts["Viewership (millions)"], model="additive", period=12)

    trend_s = decomposition.trend.dropna()
    seas_s = decomposition.seasonal.dropna()
    resid_s = decomposition.resid.dropna()

    exec_oneliner, actions = summarize_and_recommend(group_name, trend_s, seas_s, resid_s, group_ts["Viewership (millions)"])
    st.markdown(exec_oneliner)
    st.markdown("**Recommended actions:**")
    st.markdown("\n".join([f"- {a}" for a in actions]))
    st.markdown("---")

    corr_df = pd.concat([trend_s, seas_s, resid_s], axis=1).dropna()
    corr_ts = corr_df.iloc[:, 0].corr(corr_df.iloc[:, 1]) if not corr_df.empty else float("nan")
    corr_tr_res = corr_df.iloc[:, 0].corr(corr_df.iloc[:, 2]) if not corr_df.empty else float("nan")

    summary_df = pd.DataFrame({
        "Trend Mean": [trend_s.mean()],
        "Trend Std": [trend_s.std()],
        "Seasonality Mean": [seas_s.mean()],
        "Seasonality Std": [seas_s.std()],
        "Residual Mean": [resid_s.mean()],
        "Residual Std": [resid_s.std()],
        "Correlation(Trend, Seasonality)": [corr_ts],
        "Correlation(Trend, Residual)": [corr_tr_res]
    })

    summary_text = (
        f"📊 **{group_name}**\n\n"
        f"The average monthly trend in viewership is approximately **{summary_df['Trend Mean'][0]:.2f} million** viewers "
        f"with a standard deviation of **{summary_df['Trend Std'][0]:.2f}**. "
        f"Seasonal effects average **{summary_df['Seasonality Mean'][0]:.2f}** "
        f"(±{summary_df['Seasonality Std'][0]:.2f}). "
        f"Trend–seasonality correlation is **{summary_df['Correlation(Trend, Seasonality)'][0]:.2f}**."
    )

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"Time Series Decomposition: {group_name}", fontsize=18, y=1.02)

    plot_data = [
        (decomposition.trend, "Trend: long-term direction of the series (smoothed)."),
        (decomposition.seasonal, "Seasonality: recurring within-year patterns."),
        (decomposition.resid, "Residuals: irregular variation not explained by trend/seasonality."),
        (group_ts["Viewership (millions)"], "Original data: observed monthly viewership totals.")
    ]

    for ax, (series, title) in zip(axs, plot_data):
        ax.plot(series)  # ✅ no forced colors (lets matplotlib default)
        ax.set_title(title, fontsize=14, loc="left", pad=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(summary_text)
    st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)









