import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.dates as mdates
import calendar
import altair as alt
import importlib
import subprocess, sys

from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide")
st.title("🎬 TV Genre Evolution Dashboard")
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
- 🎭 **Genre-specific emotional signatures** – e.g., sustained anger in Action & Adventure, disgust & anger in Comedy & Satire, balanced joy/fear arcs in Drama.  
- ⏳ **Act-level pacing patterns** that predict retention, from setup through resolution.  
- 📆 **Cultural breakpoints** when genres pivot in tone, often tied to industry shifts or societal events.  
- 🎯 **Proven emotional hooks** that historically boost next-episode viewership per genre.  

With these insights, studios, streamers, and marketers can:  
- Detect when a series’ emotional arc diverges from successful genre norms.  
- Align scripts and campaigns with the current cultural mood.  
- Position stories and promotional beats to tap into the emotions audiences are most likely to follow.  

---

💡 **In short:** This tool turns the *long arc of genre evolution* into an actionable advantage—helping ensure stories resonate both narratively and commercially **in their moment**.

---

## **Data Sources**
- 25,000+ TV scripts from closed-caption text sources.
- Nielsen ratings cited from Wikipedia and other published sources.
- Metadata from IMDb (season, year, episode, etc.).
""")


# --- Data Loading ---
@st.cache_data(show_spinner=True)
def _read_csv(path_or_bytes):
    return pd.read_csv(path_or_bytes)

def load_data():
    st.markdown("#### Data source")
    up = st.file_uploader("Upload `Merged_Df_Cleaned.csv` (or skip if it's in the repo)", type=["csv"], key="merged_csv")

    df = None

    # 1) If user uploaded a file, use it.
    if up is not None:
        df = _read_csv(up)

    # 2) Else try an explicit path from secrets or env (configure in Cloud if you want).
    if df is None:
        for candidate in (
            st.secrets.get("DATA_CSV", ""),          # Streamlit secrets
            os.environ.get("DATA_CSV", ""),          # environment variable
        ):
            if candidate and Path(candidate).exists():
                df = _read_csv(candidate)
                break

    # 3) Else try common relative repo locations.
    if df is None:
        repo_candidates = [
            Path("data") / "Merged_Df_Cleaned.csv",
            Path("./Merged_Df_Cleaned.csv"),
        ]
        for p in repo_candidates:
            if p.exists():
                df = _read_csv(p)
                break

    # 4) If still not found, stop with a friendly message.
    if df is None:
        st.error(
            "Could not find the dataset.\n\n"
            "➡️ Upload it above **or** add a path in **Secrets** (`DATA_CSV`) or env var `DATA_CSV`, "
            "or commit it to `data/Merged_Df_Cleaned.csv` in the repo."
        )
        st.stop()

    # --- light cleaning ---
    df = df.dropna()
    df["Air Date"] = pd.to_datetime(df["Air Date"], errors="coerce")

    return df

combined_df = load_data()

# Dropdown to select a show
selected_show = st.selectbox("Select a TV Show", sorted(combined_df['Show'].dropna().unique()))
picked_show = selected_show

# Choose analysis scope
scope = st.radio("Scope", ["All data", "Selected show"], horizontal=True)

# Helper: return the DataFrame to use everywhere below
def get_scope_df(df):
    if scope == "Selected show":
        sdf = df[df["Show"] == selected_show].copy()
        if sdf.empty:
            st.warning("No rows for this show. Switching to All data.")
            return df
        return sdf
    return df

st.markdown("## Appendix 1. NLP Dictionary")
st.caption("Look up any term by category or search. Terms mirror coding used in the dashboard.")

# ---- 1) Define your glossary (category -> term -> definition) ----
GLOSSARY = {
    "Story Act Designations": {
        "Act suffixes (_1, _2, _3)":
            "Acts 1, 2, and 3 within a TV episode, determined by script length "
            "to study how emotions vary across structure."
    },
    "Emotion & Psychological Engagement": {
        "Sd_Scaled":
            "[Mean/Scaled] Scaled standard deviation of overall emotion scores across acts. "
            "Derived from the sum of per-act SDs and divided by the mean to compare variability "
            "and intensity across segments; conceptually linked to transportation (speed/depth of immersion).",
        "Anger": "[Percent] Percentage of words in an act expressing anger.",
        "Surprise": "[Percent] Percentage of words in an act expressing surprise.",
        "Disgust": "[Percent] Percentage of words in an act expressing disgust.",
        "Sadness": "[Percent] Percentage of words in an act expressing sadness.",
        "Neutral": "[Percent] Percentage of words in an act expressing neutrality.",
        "Fear": "[Percent] Percentage of words in an act expressing fear.",
        "Joy": "[Percent] Percentage of words in an act expressing joy.",
        "Positive": "[Percent] Percentage of words in an act expressing positive sentiment.",
        "Engaged":
            "[Mean] Composite score of psychological involvement or emotional investment; "
            "linked to pronouns, emotion words, and cognitive processing markers."
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
        "Cogprocess": "[Mean] Average proportion of words signaling active information processing and causation.",
        "Insight": "[Mean] Average proportion of words indicating realizations or understanding.",
        "Cause": "[Mean] Average proportion of words signaling causal relations.",
        "Discrep": "[Mean] Average proportion of counterfactual/contrastive words (e.g., should, could).",
        "Tentative": "[Mean] Average proportion of uncertainty/possibility words (e.g., maybe, perhaps).",
        "Certain": "[Mean] Average proportion of absolute words (e.g., always, never).",
        "Differ": "[Mean] Average proportion of differentiation/contrast words (e.g., but, else)."
    },
    "Perceptual Processing": {
        "Perceptual": "[Mean] Average proportion of perception words (e.g., look, heard, feel).",
        "See": "[Mean] Average proportion of visual perception words.",
        "Hear": "[Mean] Average proportion of auditory perception words.",
        "Feel": "[Mean] Average proportion of tactile/embodied sensation words."
    },
    "Motivational Drives": {
        "Drives": "[Mean] Average proportion of motivational expressions.",
        "Affiliation": "[Mean] Average proportion of social-relationship words (e.g., ally, friend).",
        "Achieve": "[Mean] Average proportion of success/accomplishment words.",
        "Power": "[Mean] Average proportion of dominance/hierarchy words.",
        "Reward": "[Mean] Average proportion of benefit/prize words.",
        "Risk": "[Mean] Average proportion of danger/uncertainty words."
    },
    "Spatial & Temporal": {
        "Relativ (Relativity)": "[Mean] Average proportion of spatial relation words (e.g., area, bend, exit).",
        "Motion": "[Mean] Average proportion of movement words (e.g., arrive, go, car).",
        "Space": "[Mean] Average proportion of spatial direction/location words (e.g., down, in).",
        "Time": "[Mean] Average proportion of temporal words (e.g., end, until, season)."
    },
}

# ---- 2) Build a flat index for search ----
flat_terms = []
for cat, terms in GLOSSARY.items():
    for term, desc in terms.items():
        flat_terms.append({"category": cat, "term": term, "definition": desc})

# ---- 3) Search box (optional) ----
q = st.text_input("Search terms (e.g., 'anger', 'WPS', 'transportation')").strip().lower()

if q:
    matches = [t for t in flat_terms if q in t["term"].lower() or q in t["definition"].lower()]
    if matches:
        # show a quick pick list of matched terms
        options = [f"{m['term']}  —  {m['category']}" for m in matches]
        sel = st.selectbox("Matching terms", options)
        picked = matches[options.index(sel)]
        st.markdown(f"### {picked['term']}")
        st.caption(picked["category"])
        st.markdown(picked["definition"])
    else:
        st.info("No matches found. Try a different keyword.")
else:
    # ---- 4) Category -> term dropdowns ----
    cat = st.selectbox("Category", list(GLOSSARY.keys()))
    term = st.selectbox("Term", list(GLOSSARY[cat].keys()))
    st.markdown(f"### {term}")
    st.caption(cat)
    st.markdown(GLOSSARY[cat][term])

# ---- 5) Optional: collapsible full list ----
with st.expander("Show full dictionary"):
    for cat, terms in GLOSSARY.items():
        st.markdown(f"**{cat}**")
        for term, desc in terms.items():
            st.markdown(f"- **{term}** — {desc}")

# ==== Section 0: Descriptive Stats for Emotion Features ====
st.header("🧮 Descriptive Statistics for Emotion & Linguistic Features")
data = get_scope_df(combined_df)

# ---- 1) Define feature prefixes + common synonyms (handles mixed naming) ----
base_prefixes = [
    'Anger','Surprise','Disgust','Sadness','Neutral','Fear','Joy',
    'Positive','Negative','Engaged','Not engaged',
    'WC','Analytic','Clout','Authentic','Tone','WPS',
    'Sixltr','Six_Let',              # both spellings show up in your study/tables
    'Dic','Dictionary',              # both spellings
    'Cogproc','Insight','Cause','Discrep','Tentat','Certain','Differ',
    'Percept','See','Hear','Feel','Drives','Affiliation','Achieve',
    'Power','Reward','Risk','Relativ','Motion','Space','Time',
    'Transportation'                 # appears in later tables
]

# build a normalized lookup so Six_Let and Sixltr map together, same for Dic/Dictionary
synonym_map = {
    'Six_Let': 'Sixltr',
    'Dictionary': 'Dic'
}

def normalize_prefix(p):
    return synonym_map.get(p, p)

# ---- 2) Collect columns that start with any prefix (e.g., Anger_1, Anger_2, Anger_3) ----
cols_in_df = list(map(str, data.columns))
candidate_cols = []
for col in cols_in_df:
    for p in base_prefixes:
        if col.startswith(p):
            candidate_cols.append(col)
            break

# de-duplicate and sort for nice display
candidate_cols = sorted(set(candidate_cols), key=lambda x: (x.split('_')[0], x))

# ---- 3) Guardrails: bail gracefully if nothing matches ----
if not candidate_cols:
    st.info("No emotion/linguistic columns detected by prefix. Showing all columns so you can verify naming.")
    st.write(cols_in_df)
else:
    # ---- 4) Overall describe across all matched features ----
    st.subheader("Overall (All Acts Combined)")
    st.caption("Includes any column whose name starts with an emotion/linguistic prefix (e.g., `Anger_1`, `WC_2`, `Dic_3`).")
    st.dataframe(data[candidate_cols].describe().transpose())

    # ---- 5) Act-specific summaries (e.g., *_1, *_2, *_3) if present ----
    for act in ('_1', '_2', '_3'):
        act_cols = [c for c in candidate_cols if c.endswith(act)]
        if act_cols:
            st.subheader(f"Act-level Summary: Act {act[-1]}")
            st.caption("Descriptive stats for features measured within this act.")
            st.dataframe(data[act_cols].describe().transpose())

    # ---- 6) Collapsed-by-prefix view (mean across acts per feature family), if you want a quick scan ----
    # build a tidy mapping: base feature name -> its columns across acts
    from collections import defaultdict
    family = defaultdict(list)
    for c in candidate_cols:
        base = c.split('_')[0]
        base = normalize_prefix(base)
        family[base].append(c)

    # compute a quick collapsed mean across acts per row, then describe
    collapsed = {}
    for base, cols in family.items():
        # only use numeric columns and handle missing safely
        valid = [c for c in cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
        if valid:
            collapsed[base] = data[valid].mean(axis=1)

    if collapsed:
        collapsed_df = pd.DataFrame(collapsed)
        st.subheader("Collapsed Feature Families (Mean Across Acts)")
        st.caption("Each column is the row-wise mean of that feature family across available acts (e.g., mean(Anger_1, Anger_2, Anger_3)).")
        st.dataframe(collapsed_df.describe().transpose())

# =======================
# 🧪 Causality (Continuous Only, robust T selection)
# =======================

st.markdown("---")
st.subheader("🧪 Causality (Double ML & Causal Forest) — Continuous Only")
st.caption("Estimate treatment effects of a selected, actionable NLP feature on next-episode viewership.")

# 0) Lazy-import econml so the app can still load if the lib isn't present
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

# 1) Basic guards / cleanup for the selected season scope
if "season_df" not in st.session_state:
    st.info("No season-level data available yet.")
    # Don't block the whole app
else:
    season_df = st.session_state["season_df"].copy()
    # De-duplicate any repeated columns (can happen after merges)
    dupes = season_df.columns[season_df.columns.duplicated()].unique()
    if len(dupes) > 0:
        st.warning(f"Duplicate columns removed: {list(map(str, dupes))}")
        season_df = season_df.loc[:, ~season_df.columns.duplicated()].copy()

    if TARGET_COL not in season_df.columns:
        st.info(f"Target column '{TARGET_COL}' is missing in the season data.")
    else:
        # 2) Candidate NLP columns by prefix
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
            or (isinstance(c, str)   and any(str(c).startswith(p)    for p in nlp_prefixes))
        ]

        # 3) UI — treatment and controls
        with st.expander("Define treatment feature (continuous z-score)", expanded=True):
            if isinstance(season_df.columns, pd.MultiIndex):
                top_level_names = sorted({c[0] for c in season_df.columns})
                t_feat = st.selectbox(
                    "Choose treatment (top level)",
                    [c for c in top_level_names if any(str(c).startswith(p) for p in nlp_prefixes)]
                )
            else:
                t_feat = st.selectbox("Choose treatment", nlp_candidates or ["(no NLP found)"])

        with st.expander("Controls (X) and model settings", expanded=True):
            default_ctrl = ["Season","Episode","WC","WPS","Analytic","Clout","Tone","Year","Time_in_minutes"]

            def _is_treatment_col(col):
                if isinstance(col, tuple):
                    return col[0] == t_feat
                return col == t_feat

            ctrl_pool = [c for c in default_ctrl if c in season_df.columns]
            extra = [c for c in season_df.columns if (c not in ctrl_pool) and not _is_treatment_col(c)]
            ctrl_pool = [c for c in ctrl_pool if isinstance(c, str)]
            extra_str = [c for c in extra if isinstance(c, str)]
            ctrl_candidates = ctrl_pool + extra_str

            X_cols = st.multiselect("Select controls (keep parsimonious)", ctrl_candidates, default=ctrl_candidates[:6])

            do_run_dml = st.checkbox("Run Double ML (LinearDML)", value=True, disabled=not econml_ok)
            do_run_cf  = st.checkbox("Run Causal Forest (CausalForestDML)", value=True, disabled=not econml_ok)

        # If econml is missing, stop this section (but continue the rest of the app)
        if not econml_ok:
            st.warning("Install/enable econml to run this section.")
        else:
            # 4) Helper: pick ONE treatment Series even with MultiIndex/duplicates
            def pick_treatment_series(df, t_feat_name):
                if isinstance(df.columns, pd.MultiIndex):
                    subs = [col for col in df.columns if col[0] == t_feat_name]
                    if len(subs) == 0:
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

            # 5) Build Y, T (z-score), X
            ready = False
            if t_feat and len(X_cols) > 0:
                # If MultiIndex, selecting t_feat alone won’t bring subcolumns; fetch Series via helper
                if not isinstance(season_df.columns, pd.MultiIndex):
                    cols_for_work = [TARGET_COL, t_feat] + X_cols
                    work = season_df[cols_for_work].copy()
                else:
                    work = season_df[[TARGET_COL] + X_cols].copy()

                work = work.apply(pd.to_numeric, errors="coerce", axis=0)

                if isinstance(work.columns, pd.MultiIndex):
                    # Target from the original df
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

                    mu = float(np.nanmean(t_raw)); sd = float(np.nanstd(t_raw))
                    T = (t_raw - mu) / (sd + 1e-9)

                    if len(Y) == len(T) == X.shape[0] and X.shape[0] >= 30 and X.shape[1] >= 2:
                        ready = True
                    else:
                        st.info(
                            f"Not enough aligned data (rows={X.shape[0] if 'X' in locals() else 0}, "
                            f"features={X.shape[1] if 'X' in locals() else 0}). Need ≥30 rows and ≥2 features."
                        )
                else:
                    st.info("Pick a valid treatment column.")
            else:
                st.info("Select a treatment and at least one control to enable modeling.")

            # 6) Run Double ML (LinearDML)
            ate = lb = ub = None
            if ready and do_run_dml:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import LassoCV

                    # Flatten shapes defensively
                    if Y.ndim == 2 and Y.shape[1] == 1: Y = Y.ravel()
                    if T.ndim == 2 and T.shape[1] == 1: T = T.ravel()
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

                        # Policy table
                        st.markdown("**Policy Rule (Continuous): Aim for +k SD increase**")
                        k = st.slider("Target increase (k SD)", 0.1, 1.0, 0.5, 0.1, key="k_sd_dml")
                        target_raw = mu + k * sd
                        cur = t_raw
                        gap_raw = np.maximum(0.0, target_raw - cur)
                        gap_sd  = np.minimum(k, gap_raw / (sd + 1e-9))
                        expected_lift = ate * gap_sd

                        rule_df = pd.DataFrame({
                            "Row": aligned.index,
                            f"{t_feat}_raw": cur,
                            f"{t_feat}_z": (cur - mu) / (sd + 1e-9),
                            "target_raw": target_raw,
                            "raise_by_raw": gap_raw,
                            "raise_by_SD": gap_sd,
                            "expected_lift": expected_lift
                        }).sort_values("expected_lift", ascending=False)
                        st.dataframe(rule_df, use_container_width=True)

                except Exception as e:
                    st.error(f"LinearDML failed: {e}")

            # 7) Causal Forest (CATE)
            cate = coverage = policy_value = None
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
                        st.caption("Positive CATE ⇒ episodes predicted to benefit more from increasing the treatment.")

                        q = st.slider("Target top q% episodes by predicted CATE", 5, 50, 20, step=5, key="q_cf")
                        thr = np.nanpercentile(cate, 100 - q)
                        take = cate >= thr
                        coverage = float(np.mean(take))
                        policy_value = float(np.nanmean(cate[take])) if np.any(take) else float("nan")

                        c1, c2 = st.columns(2)
                        c1.metric("Coverage", f"{coverage*100:.1f}%")
                        c2.metric("Avg predicted lift (targeted)", f"{policy_value:.4f}")

                except Exception as e:
                    st.error(f"Causal Forest failed: {e}")

            # 8) Narrative summary
            def _fmt(x, nd=4):
                try: return f"{float(x):.{nd}f}"
                except Exception: return "—"

            def interpret_ate(ate_, lb_, ub_, t_feat_name, target_name):
                if ate_ is None or lb_ is None or ub_ is None:
                    return "Run Double ML to estimate the overall effect (ATE)."
                if lb_ > 0:
                    return (f"Increasing **{t_feat_name}** likely **increases** **{target_name}** on average "
                            f"(ATE {ate_:.4f} per +1 SD; 95% CI [{lb_:.4f}, {ub_:.4f}]).")
                elif ub_ < 0:
                    return (f"Increasing **{t_feat_name}** likely **decreases** **{target_name}** on average "
                            f"(ATE {ate_:.4f} per +1 SD; 95% CI [{lb_:.4f}, {ub_:.4f}]).")
                else:
                    return (f"The **overall effect** of **{t_feat_name}** on **{target_name}** is **inconclusive** "
                            f"(ATE {ate_:.4f} per +1 SD; 95% CI [{lb_:.4f}, {ub_:.4f}] spans 0).")

            st.markdown("### 📗 What the terms mean")
            controls_str = ", ".join([c for c in X_cols if c != t_feat]) or "(none)"
            st.markdown(
                f"- **Treatment (T)** — **{t_feat}** as a z-score (+1 SD = one standard deviation).\n"
                f"- **Outcome (Y)** — **{TARGET_COL}**.\n"
                f"- **Controls (X)** — {controls_str}.\n"
                f"- **ATE** — average effect of +1 SD in **{t_feat}** across all episodes.\n"
                f"- **CATE** — episode-specific effect given its context X."
            )

            st.markdown("### 🌍 Overall effect (ATE)")
            st.markdown(interpret_ate(ate, lb, ub, t_feat, TARGET_COL))
            if ate is not None:
                st.caption("Scale the ATE by your feasible change k SD (e.g., ×0.5 if you can move half a SD).")

            st.markdown("### 🎯 Who to target (CATE-based policy)")
            if cate is not None and coverage is not None and policy_value is not None:
                mean_cate = float(np.nanmean(cate)) if hasattr(np, "nanmean") else None
                st.markdown(
                    f"Target the **top {int(st.session_state.get('q_cf', 20))}%** by predicted CATE. "
                    f"Coverage: **{_fmt(coverage*100,1)}%**. "
                    f"Predicted gain (per +1 SD): **{_fmt(policy_value)}**."
                )
                if mean_cate is not None and mean_cate < 0 and policy_value > 0:
                    st.info(
                        "Mean CATE can be negative overall while targeted gain is positive — "
                        "because we cherry-pick the episodes with positive predicted effects."
                    )
            else:
                st.markdown("Run **Causal Forest** to enable CATE targeting.")
         
# =======================
# 🔎 Automated Narrative Summary (continuous-only, plain-English)
# =======================

def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def interpret_ate(ate, lb, ub, t_feat, target_name):
    """Significance-aware, human-friendly ATE sentence (per +1 SD)."""
    if lb > 0:
        return (f"Increasing **{t_feat}** likely **increases** **{target_name}** on average "
                f"(ATE {ate:.4f} per +1 SD; 95% CI [{lb:.4f}, {ub:.4f}]).")
    elif ub < 0:
        return (f"Increasing **{t_feat}** likely **decreases** **{target_name}** on average "
                f"(ATE {ate:.4f} per +1 SD; 95% CI [{lb:.4f}, {ub:.4f}]).")
    else:
        return (f"The **overall effect** of **{t_feat}** on **{target_name}** is **inconclusive** "
                f"(ATE {ate:.4f} per +1 SD; 95% CI [{lb:.4f}, {ub:.4f}] spans 0).")

controls_str = ", ".join([c for c in X_cols if c != t_feat]) or "(none)"
ate_available = ('ate' in locals()) and ('lb' in locals()) and ('ub' in locals())
cf_available  = ('cate' in locals()) and ('coverage' in locals()) and ('policy_value' in locals()) and ('q' in locals())

# --- Plain-English definitions ---
st.markdown("### 📗 What the terms mean (plain English)")
st.markdown(f"""
- **Treatment (T)** — the knob we imagine turning: **{t_feat}** (encoded as a **z-score**, so “+1 SD” means one standard deviation higher than the mean).
- **Outcome (Y)** — what we want to move: **{TARGET_COL}**.
- **Controls (X)** — context we adjust for (to make apples-to-apples): **{controls_str}**.
- **ATE (Average Treatment Effect)** — the **average** change in **{TARGET_COL}** if we increase **{t_feat}** by **+1 SD** across **all episodes**.
- **CATE (Conditional Average Treatment Effect)** — the **episode-specific** effect; for each episode (given its context X), the predicted change in **{TARGET_COL}** for **+1 SD** in **{t_feat}**.
- **Coverage** — the **share of episodes** we choose to act on with a simple rule (e.g., target the **top {int(q) if 'q' in locals() else 'q'}%** by predicted CATE → coverage is about that percent; ties can shift it slightly).
- **Predicted lift (targeted)** — among the **episodes we target**, the **average predicted improvement** in **{TARGET_COL}** for a **+1 SD** increase in **{t_feat}**.  
  If we can only change by **k SD** (e.g., k = 0.5), expected gain ≈ **predicted lift × k**.
""")

# --- Setup line ---
st.markdown("### 🧾 What we ran")
st.markdown(
    f"We treated **{t_feat}** as **continuous (z-score)** and adjusted for **{controls_str}**. "
    "We estimated the overall effect (ATE) using **Double ML (LinearDML)** and the episode-level effects (CATE) using a **Causal Forest**."
)

# --- ATE line ---
st.markdown("### 🌍 Overall effect (ATE)")
if ate_available:
    st.markdown(interpret_ate(ate, lb, ub, t_feat, TARGET_COL))
    st.caption("Reading: “per +1 SD” means nudging the treatment by one standard deviation. "
               "If you can only move it by k SD (e.g., k = 0.5), multiply the ATE by k for a rough expected change.")
else:
    st.markdown("Run Double ML to estimate the **overall effect (ATE)**.")

# --- CATE / Targeting ---
st.markdown("### 🎯 Who to target (CATE-based policy)")
if cf_available:
    mean_cate = float(np.nanmean(cate)) if hasattr(np, "nanmean") else None
    st.markdown(
        f"We ranked episodes by predicted **CATE** and targeted the **top {int(q)}%** (those predicted to benefit most). "
        f"**Coverage:** {_fmt(coverage*100,1)}% of episodes. "
        f"**Predicted gain (targeted, per +1 SD):** {_fmt(policy_value)}."
    )
    st.caption("This policy focuses effort where the model predicts benefit. "
               "Scale gains by your feasible change **k SD** (e.g., ×0.5 if k = 0.5).")
    # Optional clarity if mean CATE is negative but targeted lift is positive
    if mean_cate is not None and mean_cate < 0 and policy_value > 0:
        st.info(
            "Why is **mean CATE negative** while **predicted gain for targeted episodes is positive**? "
            "Because we **select the winners**. The mean CATE averages everyone (many episodes look slightly negative), "
            "but the targeting rule cherry-picks the episodes with **positive** predicted effects."
        )
else:
    st.markdown("Run **Causal Forest** to enable **CATE**, targeting, coverage, and predicted gains.")

# --- Managerial takeaway ---
st.markdown("### 🧭 Managerial takeaway")
if ate_available:
    if lb > 0:
        st.markdown(
            f"- Evidence suggests raising **{t_feat}** tends to **increase** **{TARGET_COL}** on average.\n"
            f"- Still, prioritize episodes with **high predicted CATE** (the top-{int(q) if 'q' in locals() else 'q'}% rule) "
            "and use the rule table to set a feasible **k SD** adjustment."
        )
    elif ub < 0:
        st.markdown(
            f"- Evidence suggests raising **{t_feat}** tends to **decrease** **{TARGET_COL}** on average.\n"
            f"- Avoid blanket increases; if you act, focus only on episodes with **clearly positive** predicted CATE and realistic **k SD** changes."
        )
    else:
        st.markdown(
            "- The **overall effect is inconclusive**. Avoid blanket changes.\n"
            f"- Use **CATE targeting** to focus on episodes where the model predicts a **positive** gain, "
            "and set feasible **k SD** adjustments via the rule table."
        )
else:
    st.markdown(
        "- Run **Double ML** to get the overall direction (ATE), then use **CATE** targeting to guide where to act.\n"
        "- Always sanity-check feasibility (how many SD you can realistically move) and costs."
    )


#### Time Decomposition ####  
st.header("📉 Time Series Decomposition")

st.markdown("""
**How to read these components**
- **Trend:** Long-term direction of viewership after smoothing out seasonality and noise.
- **Seasonality:** Recurring within-year patterns (e.g., fall premieres, holidays).
- **Residuals:** Irregular, unexplained variation not captured by trend/seasonality.
- **Original Data:** Observed monthly totals for the genre group.
""")

# ---------- Helper: auto-summarize + exec TL;DR + actions ----------
def _classify(value, thresholds, labels):
    """Map a numeric value to a label via ascending thresholds."""
    for t, lab in zip(thresholds, labels):
        if value <= t:
            return lab
    return labels[-1]

def _month_abbr_list(month_indices):
    return [calendar.month_abbr[int(m)] for m in month_indices]

def summarize_and_recommend(group_name, trend_s, seas_s, resid_s, orig_series):
    """Create plain-English summary, exec one-liner, and recommended actions."""
    tr = trend_s.dropna()
    if len(tr) >= 3:
        # slope per year
        x_years = (tr.index.view('int64') / 1e9 / 86400 / 365.25)
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

    trend_vol = tr.std()
    trend_vol_label = _classify(trend_vol, [0.25, 0.75, 1.5], ["very low", "low", "moderate", "high", "very high"])

    ss = seas_s.dropna()
    seas_amp = (ss.quantile(0.95) - ss.quantile(0.05))
    orig_sd = orig_series.std()
    seas_strength_ratio = 0 if orig_sd == 0 else (seas_amp / max(orig_sd, 1e-9))
    seas_strength_label = _classify(seas_strength_ratio, [0.1, 0.25, 0.5], ["weak", "mild", "moderate", "strong"])

    if len(ss) > 0:
        month_means = ss.groupby(ss.index.month).mean().sort_values(ascending=False)
        top_months = _month_abbr_list(list(month_means.index[:2]))
    else:
        top_months = []

    rs = resid_s.dropna()
    resid_sd = rs.std()
    resid_vol_label = _classify(resid_sd, [0.15, 0.35, 0.75], ["very low", "low", "moderate", "high", "very high"])

    aligned_corr = np.nan
    if len(tr) > 0 and len(ss) > 0:
        corr_df = pd.concat([tr, ss], axis=1, join='inner').dropna()
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

    months_txt = ", ".join(top_months) if top_months else "—"
    summary_md = f"""
**{group_name}: What to know at a glance**

- {slope_emoji} **Trend:** **{trend_dir}** over time; long-run volatility is **{trend_vol_label}**.
- 🗓️ **Seasonality:** **{seas_strength_label}**; best months: **{months_txt}**.
- 🔀 **Residuals:** **{resid_vol_label}** (watch for one-off spikes/drops).
- 🔗 **Trend vs. Seasonality:** **{corr_label}** (corr≈{aligned_corr:.2f}).
""".strip()

    return summary_md, exec_oneliner, actions

# ---------- Clean & bound dates ----------
data = get_scope_df(combined_df)
data["Air Date"] = pd.to_datetime(data["Air Date"], errors="coerce")
data = data.dropna(subset=["Air Date"])
today = pd.Timestamp.today().normalize()
data = data[(data["Air Date"] >= pd.Timestamp("1985-01-01")) & (data["Air Date"] <= today)]
data = data.set_index("Air Date").sort_index()

genre_groups = {
    "Comedy & Satire": ["Sitcom", "Comedy-Drama", "Animated sitcom", "Comedy", "Comedy horror"],
    "Drama": ["Drama", "Medical drama", "Family drama", "Period drama", "Political drama", "Psychological drama", "Serial drama"],
    "Crime, Law & Justice": ["Crime drama", "Police procedural", "Legal drama", "Legal thriller", "Crime", "Psychological thriller"],
    "Action & Adventure": ["Action", "Action-adventure", "Superhero", "Adventure", "Action fiction"],
    "Fantasy & Sci-Fi": ["Fantasy", "Science fiction", "Science fantasy", "Supernatural", "Supernatural drama"],
    "Horror & Suspense": ["Horror", "Mystery", "Psychological horror", "Thriller"]
}

# ---------- Loop genre groups ----------
for group_name, subgenres in genre_groups.items():
    group_df = data[data["Genre"].isin(subgenres)]
    if group_df.empty:
        continue

    # Monthly totals; basic cleaning
    group_ts = group_df[["Viewership (millions)"]].resample("M").sum()
    group_ts = group_ts[group_ts["Viewership (millions)"] > 0]
    group_ts = group_ts.ffill(limit=2)
    group_ts = group_ts[group_ts.index <= today]

    if group_ts.shape[0] < 24:
        continue  # need at least 2 years for decomposition

    # Decompose
    decomposition = seasonal_decompose(group_ts["Viewership (millions)"], model="additive", period=12)

    # NaN-safe series
    trend_s = decomposition.trend.dropna()
    seas_s  = decomposition.seasonal.dropna()
    resid_s = decomposition.resid.dropna()

    # ---- Exec TL;DR + actions FIRST ----
    summary_md, exec_oneliner, actions = summarize_and_recommend(
        group_name, trend_s, seas_s, resid_s, group_ts["Viewership (millions)"]
    )
    st.markdown(exec_oneliner)
    st.markdown("**Recommended actions:**")
    st.markdown("\n".join([f"- {a}" for a in actions]))
    st.markdown("---")

    # ---- Numeric summary (NaN-safe) ----
    corr_df_ts = pd.concat([trend_s, seas_s, resid_s], axis=1).dropna()
    corr_ts_ts = corr_df_ts.iloc[:, 0].corr(corr_df_ts.iloc[:, 1]) if not corr_df_ts.empty else float('nan')
    corr_tr_res = corr_df_ts.iloc[:, 0].corr(corr_df_ts.iloc[:, 2]) if not corr_df_ts.empty else float('nan')

    summary_df = pd.DataFrame({
        'Trend Mean': [trend_s.mean()],
        'Trend Std': [trend_s.std()],
        'Seasonality Mean': [seas_s.mean()],
        'Seasonality Std': [seas_s.std()],
        'Residual Mean': [resid_s.mean()],
        'Residual Std': [resid_s.std()],
        'Correlation(Trend, Seasonality)': [corr_ts_ts],
        'Correlation(Trend, Residual)': [corr_tr_res]
    })

    summary_text = (
        f"📊 **{group_name}**\n\n"
        f"The average monthly trend in viewership is approximately **{summary_df['Trend Mean'][0]:.2f} million** viewers "
        f"with a standard deviation of **{summary_df['Trend Std'][0]:.2f}**. "
        f"Seasonal effects average **{summary_df['Seasonality Mean'][0]:.2f}** "
        f"(±{summary_df['Seasonality Std'][0]:.2f}). "
        f"Trend–seasonality correlation is **{summary_df['Correlation(Trend, Seasonality)'][0]:.2f}**."
    )

    # ---- Plot (bold titles + every-2-years ticks on each subplot) ----
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"Time Series Decomposition: {group_name}", fontsize=18, y=1.02)

    plot_data = [
        (decomposition.trend, '#1f77b4', "**Trend**\nLong-term direction of the series, smoothing out seasonal effects and noise."),
        (decomposition.seasonal, '#2ca02c', "**Seasonality**\nRecurring patterns within each year (e.g., premieres, holidays)."),
        (decomposition.resid, '#d62728', "**Residuals**\nIrregular, unexplained variation not captured by trend or seasonality."),
        (group_ts["Viewership (millions)"], '#9467bd', "**Original Data**\nObserved monthly viewership totals for this genre group.")
    ]

    for ax, (series, color, title) in zip(axs, plot_data):
        ax.plot(series, color=color)
        ax.set_title(title, fontsize=14, loc='left', pad=10, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    plt.tight_layout()
    st.pyplot(fig)

    # ---- Numeric text + table last ----
    st.markdown(summary_text)
    st.dataframe(summary_df.style.format("{:.4f}"))


