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
from pathlib import Path

@st.cache_data(show_spinner=False)
def load_data():
    # Prefer relative path so it works on Streamlit Cloud
    path = Path(__file__).parent / "Merged_Df_Cleaned.csv"

    # Fallback to manual upload if missing or still an LFS pointer
    if not path.exists() or path.stat().st_size < 500:  # pointer files are tiny
        up = st.file_uploader("Upload Merged_Df_Cleaned.csv", type=["csv"])
        if up is None:
            st.stop()
        df = pd.read_csv(up, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

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
# 🎬 Series/Season Filters + 🧾 NLP Metrics (Series & Season) + 🌟 Season Feature Selection
# =======================

# ---- Load & clean base data ----
data = get_scope_df(combined_df).copy()
data.columns = data.columns.str.strip()

# ---- Guard: need a Show column for dropdowns ----
if "Show" not in data.columns:
    st.error("Column 'Show' not found in the dataframe.")
    st.write("DEBUG columns:", list(data.columns))  # remove after debugging
    st.stop()

# =======================
# 🎬 Series & 📆 Season dropdowns (cascading)
# =======================
series_opts = sorted([str(x) for x in data["Show"].dropna().unique().tolist()])

series_df = data.loc[data["Show"] == picked_show].copy()
if series_df.empty:
    st.info("No rows found for the selected TV series.")
    st.stop()

# Season (optional, restricted to the chosen series)
if "Season" in series_df.columns and not series_df["Season"].dropna().empty:
    season_opts = sorted(series_df["Season"].dropna().unique().tolist())
    # Map labels to raw values (handles numeric/int seasons)
    season_label_map = {str(v): v for v in season_opts}
    picked_season_label = st.selectbox(
        "📆 Select season (optional)",
        ["-- All seasons --"] + list(season_label_map.keys()),
        key="season_select_allin1"
    )
    if picked_season_label != "-- All seasons --":
        picked_season = season_label_map[picked_season_label]
        season_df = series_df.loc[series_df["Season"] == picked_season].copy()
    else:
        picked_season = "-- All seasons --"
        season_df = series_df.copy()
else:
    picked_season = "-- All seasons --"
    season_df = series_df.copy()

# Save for downstream use (optional)
st.session_state["picked_show"] = picked_show
st.session_state["picked_season"] = picked_season
st.session_state["series_df"] = series_df
st.session_state["season_df"] = season_df

# =======================
# 🌟 Gradient Boosting + SHAP with Definitions (Series & Season)
# =======================
st.markdown("---")
st.subheader("🌟 Top 10 Predictive Features with Definitions")

TARGET_COL = "Viewership (millions)"

# --- Helpers ---

def _drop_cols(df, target):
    return [
        "Show","Cancelled","Season","Episode","Time","Air Date","Air_Date","AirDate","Date",
        target,"Genre","Aggregate Genre","Year","No.of seasons","Time_in_minutes",
        "Episode Length","No.of episodes","Not engaged_1","Not engaged_2","Not engaged_3"
    ] + [c for c in df.columns if str(c).startswith("Network_")]

def _build_xy(df, target):
    X = df.drop(columns=_drop_cols(df, target), errors="ignore").apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df.get(target), errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    X = X.dropna(axis=1, how="all").fillna(0)
    return X, y

# --- add this helper near your other helpers (above _metrics_block) ---
def _rmse_compat(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        # Newer sklearn
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Older sklearn (no 'squared' kwarg)
        return np.sqrt(mean_squared_error(y_true, y_pred))

def _metrics_block(y_true, y_pred, scope_label):
    from sklearn.metrics import r2_score, mean_absolute_error
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = _rmse_compat(y_true, y_pred)   # <-- use helper here
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{scope_label} R² (test)", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")

def _feature_base_name(col_name: str) -> str:
    base = str(col_name).split('_')[0]
    synonyms = {"Six_Let": "Sixltr", "Dictionary": "Dic"}
    return synonyms.get(base, base)

def _lookup_definition(base_name: str) -> str:
    for cat, terms in GLOSSARY.items():
        if base_name in terms:
            return terms[base_name]
    return "Definition not found in glossary."

def _render_definitions_block(title: str, feature_list, max_items: int = 30):
    bases = [_feature_base_name(f) for f in feature_list]
    bases = [b for b in dict.fromkeys(bases) if b]  # unique, preserve order
    if not bases:
        return
    with st.expander(f"{title} — definitions", expanded=False):
        shown = 0
        for b in bases:
            if shown >= max_items:
                st.caption(f"Showing first {max_items} definitions.")
                break
            desc = _lookup_definition(b)
            st.markdown(f"- **{b}** — {desc}")
            shown += 1

# --- helper: make a mini paragraph for the SHAP bar plot (Top-10) ---
def _summarize_shap_bar(df_imp: pd.DataFrame, scope_label: str, glossary_n: int = 3) -> str:
    """
    df_imp must include: ['Feature','Mean |SHAP|','Mean SHAP','Direction'] for the Top-10.
    Produces a short paragraph highlighting dominant drivers, directions, and a few definitions.
    """
    if df_imp.empty:
        return f"No SHAP summary available for {scope_label}."

    total = df_imp["Mean |SHAP|"].sum()
    if total == 0 or not np.isfinite(total):
        return f"No SHAP signal detected for {scope_label}."

    d = df_imp.copy()
    d["Share"] = (d["Mean |SHAP|"] / total).fillna(0)

    lead = d.head(3).assign(SharePct=lambda t: (100 * t["Share"]).round(1))
    lead_feats = [f"**{r.Feature}** ({r.SharePct:.1f}%)" for _, r in lead.iterrows()]

    ups   = d[d["Direction"] == "↑"]["Feature"].tolist()
    downs = d[d["Direction"] == "↓"]["Feature"].tolist()
    n_up, n_down = len(ups), len(downs)
    bal = "mixed effects" if (n_up and n_down) else ("mostly upward effects" if n_up else "mostly downward effects")

    defs_bits = []
    for f in d["Feature"].head(glossary_n):
        base = _feature_base_name(f)
        desc = _lookup_definition(base)
        if desc and desc != "Definition not found in glossary.":
            defs_bits.append(f"**{base}**: {desc}")
    defs_text = (" " + " ".join(defs_bits)) if defs_bits else ""

    parts = []
    parts.append(
        f"In **{scope_label}**, the largest drivers of predicted **{TARGET_COL}** are "
        + ", ".join(lead_feats[:-1]) + (", and " if len(lead_feats) > 1 else "")
        + (lead_feats[-1] if lead_feats else "") + "."
    )

    if n_up + n_down > 0:
        if n_up and n_down:
            parts.append(
                f"Positive lift includes {', '.join(ups[:3])}{'…' if len(ups) > 3 else ''}; "
                f"negative lift includes {', '.join(downs[:3])}{'…' if len(downs) > 3 else ''}."
            )
        elif n_up:
            parts.append(f"Most Top-10 features push **upward** ({', '.join(ups[:5])}{'…' if len(ups)>5 else ''}).")
        else:
            parts.append(f"Most Top-10 features push **downward** ({', '.join(downs[:5])}{'…' if len(downs)>5 else ''}).")

    parts.append(f"Overall, contributions show **{bal}** based on average absolute SHAP across the test set.{defs_text}")
    return " ".join(parts)

def _scope_block(df_scope, scope_label: str):
    # ---- guards ----
    if df_scope.empty or TARGET_COL not in df_scope.columns:
        st.info(f"{scope_label}: no data or target column missing.")
        return

    X, y = _build_xy(df_scope, TARGET_COL)
    if X.shape[0] < 20 or X.shape[1] < 2:
        st.info(f"{scope_label}: not enough data (rows={X.shape[0]}, features={X.shape[1]}).")
        return

    # ---- model ----
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ---- metrics ----
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = _rmse_compat(y_te, y_pred)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{scope_label} R² (test)", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")

    # ---- SHAP + Top-10 ----
    try:
        import shap, numpy as np, pandas as pd, matplotlib.pyplot as plt

        bg_n = min(500, X_tr.shape[0])
        X_bg = X_tr.sample(bg_n, random_state=42) if X_tr.shape[0] > bg_n else X_tr
        explainer = shap.Explainer(model, X_bg)
        sv = explainer(X_te)  # (n_test, n_features)

        mean_abs    = np.abs(sv.values).mean(axis=0)
        mean_signed = sv.values.mean(axis=0)

        df_imp = (
            pd.DataFrame({
                "Feature": X_te.columns,
                "Mean |SHAP|": mean_abs,
                "Mean SHAP": mean_signed
            })
            .sort_values("Mean |SHAP|", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        df_imp["Direction"] = df_imp["Mean SHAP"].apply(lambda v: "↑" if v > 0 else ("↓" if v < 0 else "0"))

        # ---- table ----
        st.markdown(f"**Top 10 — {scope_label}**")
        st.dataframe(df_imp[["Feature","Mean |SHAP|","Direction","Mean SHAP"]], use_container_width=True)

        # ---- definitions expander ----
        _render_definitions_block(
            title=f"Top-10 features ({scope_label})",
            feature_list=df_imp["Feature"].tolist()
        )

        # ---- plots ----
        with st.expander(f"SHAP summary plot — {scope_label}", expanded=False):
            fig = plt.figure()
            shap.summary_plot(sv, features=X_te, feature_names=X_te.columns, show=False)
            st.pyplot(fig)

        with st.expander(f"SHAP bar plot — {scope_label}", expanded=False):
            fig = plt.figure()
            shap.plots.bar(sv, show=False, max_display=10)
            st.pyplot(fig)
            # ✅ mini paragraph directly under the bar chart
            st.markdown(_summarize_shap_bar(df_imp, scope_label), help="Auto-generated from Top-10 SHAP bars.")

    except ModuleNotFoundError:
        st.error("`shap` not installed. Run `pip install shap` and reload.")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# =======================
# 🎬 Creative "Gist" Summary (plain-English)
# =======================
# OPTIONAL: turn on/off in the UI
ENABLE_GIST_MODE = True

# Words a writer/showrunner can skim
def _friendly_feature_name(raw: str) -> str:
    # 1) map known codes to human terms (extend as you like)
    mapping = {
        "Anger":"anger/conflict beats",
        "Surprise":"twists/surprises",
        "Disgust":"moral tension/disgust",
        "Sadness":"sad/vulnerable beats",
        "Fear":"suspense/fear",
        "Joy":"joy/relief",
        "Positive":"optimistic tone",
        "Negative":"negative tone",
        "Engaged":"intense/first-person focus",
        "Analytic":"logical/analytical language",
        "Clout":"confident voice",
        "Authentic":"authentic voice",
        "Tone":"overall tone",
        "Cogproc":"thinking/processing words",
        "Insight":"insight/reflection",
        "Cause":"cause-and-effect links",
        "Discrep":"tension/contradiction words",
        "Tentat":"hedging/uncertainty",
        "Certain":"certainty/assertion",
        "Differ":"contrast/comparison",
        "See":"visual description",
        "Hear":"audio/sound cues",
        "Feel":"bodily/feel cues",
        "Time":"time markers",
        "Motion":"movement/action words",
        "Space":"place/setting words",
        "Relativ":"relational terms",
        "Power":"status/power cues",
        "Achieve":"goals/achievement cues",
        "Reward":"rewards/payoffs",
        "Risk":"risk/peril language",
        "Affiliation":"relationships/belonging",
        "WPS":"longer sentences",
        "Sixltr":"bigger words",
        "Word_Count":"more dialogue",
        "Dic":"richer vocab",
        "SD_scaled":"tone variability and transportation"
    }
    base = raw.split("_")[0]
    return mapping.get(base, raw.replace("_", " ").lower())

def _creative_hint(name: str, direction: str) -> str:
    """
    Turn (feature, ↑/↓/0) into a friendly nudge.
    Keep this punchy; we're aiming for show notes, not stats.
    """
    up = direction == "↑"
    dn = direction == "↓"

    # lightweight heuristics; tweak freely
    if "twist" in name or "surprise" in name:
        return ("Lean into a reveal earlier" if up else "Save twists for key turns") if (up or dn) else "Use reveals purposefully"
    if "anger" in name or "conflict" in name:
        return ("Channel conflict into plot stakes" if up else "Trim shouting; use sharper subtext") if (up or dn) else "Use conflict to serve stakes"
    if "joy" in name or "relief" in name:
        return ("Add moments of lift/humor" if up else "Avoid undercutting serious beats") if (up or dn) else "Balance heaviness with lift"
    if "fear" in name or "suspense" in name or "risk" in name:
        return ("Build tension and uncertainty" if up else "Don’t over-milk dread") if (up or dn) else "Modulate suspense"
    if "sad" in name or "vulnerable" in name:
        return ("Let characters feel losses" if up else "Don’t linger too long on lows") if (up or dn) else "Use vulnerability to earn payoff"
    if "confident" in name or "clout" in name:
        return ("Give leads decisive lines" if up else "Soften bravado with doubt") if (up or dn) else "Tune confidence beats"
    if "authentic" in name:
        return ("Keep dialogue grounded" if up else "Tighten to avoid meandering") if (up or dn) else "Favor honest, specific lines"
    if "visual" in name or "audio" in name or "place" in name or "movement" in name:
        return ("Show, don’t tell—stage it" if up else "Cut busy blocking; focus the frame") if (up or dn) else "Use visuals to carry story"
    if "insight" in name or "thinking" in name:
        return ("Let characters process on-screen" if up else "Trim inner monologue") if (up or dn) else "Balance thought vs action"
    if "optimistic" in name or "positive" in name:
        return ("Offer hope in resolutions" if up else "Keep endings honest, not saccharine") if (up or dn) else "Shape the aftertaste"
    if "negative tone" in name:
        return ("Let stakes feel real" if up else "Avoid relentless gloom") if (up or dn) else "Calibrate darkness"
    if "longer sentences" in name or "bigger words" in name:
        return ("Let speeches breathe" if up else "Tighten lines for pace") if (up or dn) else "Vary sentence length for rhythm"
    if "more dialogue" in name:
        return ("Write it on the page" if up else "Let silence do work") if (up or dn) else "Balance talk vs silence"
    if "rewards" in name or "payoffs" in name:
        return ("Pay off setups clearly" if up else "Delay gratification to build itch") if (up or dn) else "Mind the setup–payoff chain"

    return ("Use more of this lever" if up else "Dial this back") if (up or dn) else "Use selectively"

def _creative_gist_summary(df_imp: pd.DataFrame, scope_label: str) -> str:
    """
    Produce a one-liner + 3 bullets creatives can act on.
    Expects columns: Feature, Direction, Mean |SHAP|
    """
    if df_imp.empty:
        return f"**{scope_label} — Creative Gist**\n- No clear drivers yet."

    # order already by importance; take a few
    top = df_imp.head(5).copy()
    top["nice"] = top["Feature"].apply(_friendly_feature_name)

    # one-liner logline: up to 3 key levers with arrows
    headline_bits = [f"{row['nice']} {row['Direction']}" for _, row in top.iloc[:3].iterrows()]
    headline = " • ".join(headline_bits)

    # actionable nudges (3 bullets)
    bullets = []
    for _, row in top.iterrows():
        bullets.append(f"- **{row['nice']}**: {_creative_hint(row['nice'], row['Direction'])}")
        if len(bullets) == 3:
            break

    # do more / ease up lists
    do_more  = [r["nice"] for _, r in top[top["Direction"]=="↑"].iterrows()][:3] or ["—"]
    ease_up  = [r["nice"] for _, r in top[top["Direction"]=="↓"].iterrows()][:3] or ["—"]

    lines = []
    lines.append(f"**{scope_label} — Creative Gist**")
    lines.append(f"*What tends to play better here:* {headline}")
    lines.extend(bullets)
    lines.append(f"- **Do more of:** {', '.join(do_more)}")
    lines.append(f"- **Ease up on:** {', '.join(ease_up)}")
    return "\n".join(lines)

    # --- Train Gradient Boosting ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    _metrics_block(y_te, y_pred, scope_label)

    # --- SHAP explainability ---
    try:
        import shap, numpy as np, pandas as pd, matplotlib.pyplot as plt
        bg_n = min(500, X_tr.shape[0])
        X_bg = X_tr.sample(bg_n, random_state=42) if X_tr.shape[0] > bg_n else X_tr
        explainer = shap.Explainer(model, X_bg)
        sv = explainer(X_bg)

        mean_abs = np.abs(sv.values).mean(axis=0)
        mean_signed = sv.values.mean(axis=0)
        df_imp = (pd.DataFrame({
                    "Feature": X.columns,
                    "Mean |SHAP|": mean_abs,
                    "Mean SHAP": mean_signed
                 })
                 .sort_values("Mean |SHAP|", ascending=False)
                 .head(10)
                 .reset_index(drop=True))
        df_imp["Direction"] = df_imp["Mean SHAP"].apply(lambda v: "↑" if v > 0 else ("↓" if v < 0 else "0"))

        st.markdown(f"**Top 10 — {scope_label}**")
        st.dataframe(df_imp[["Feature","Mean |SHAP|","Direction","Mean SHAP"]], use_container_width=True)

        # Definitions under the table
        _render_definitions_block(
            title=f"Top-10 features ({scope_label})",
            feature_list=df_imp["Feature"].tolist()
        )

        with st.expander(f"SHAP summary plot — {scope_label}", expanded=False):
            fig = plt.figure()
            shap.summary_plot(sv, features=X_tr, feature_names=X.columns, show=False)
            st.pyplot(fig)

        with st.expander(f"SHAP bar plot — {scope_label}", expanded=False):
            fig = plt.figure()
            shap.plots.bar(sv, show=False, max_display=10)
            st.pyplot(fig)

    except ModuleNotFoundError:
        st.error("`shap` not installed. Run `pip install shap` and reload.")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# --- Layout: Series (left) | Season (right) ---
col1, col2 = st.columns(2)
with col1:
    _scope_block(series_df, "Series")
with col2:
    _scope_block(season_df, "Season")

# =======================
# 🤖 Modeling: Elastic Net, Gradient Boosting, XGBoost (Series Level)
# =======================
st.markdown("---")
st.subheader("🤖 Predictive Modeling (Series Scope)")

with st.expander("Train a model on the current TV series scope", expanded=True):
    TARGET_COL = "Viewership (millions)"
    if TARGET_COL not in series_df.columns:
        st.warning(f"Target column '{TARGET_COL}' not found in series data; cannot train models.")
    else:
        # Drop clear non-feature columns; keep numeric only for modeling
        drop_cols_model = [
            "Show","Cancelled","Season","Episode","Time","Air Date","Air_Date","AirDate","Date",
            TARGET_COL,"Genre","Aggregate Genre","Year","No.of seasons","Time_in_minutes",
            "Episode Length","No.of episodes","Not engaged_1","Not engaged_2","Not engaged_3"
        ] + [c for c in series_df.columns if c.startswith("Network_")]

        X_all = (
            series_df.drop(columns=drop_cols_model, errors="ignore")
                     .apply(pd.to_numeric, errors="coerce")
        )
        y_all = pd.to_numeric(series_df[TARGET_COL], errors="coerce")

        # Keep rows with a target and at least some features
        mask = y_all.notna()
        X_all, y_all = X_all.loc[mask], y_all.loc[mask]
        # Drop all-NaN columns; fill remaining NaNs with 0
        X_all = X_all.dropna(axis=1, how="all").fillna(0)

        if X_all.shape[0] < 20 or X_all.shape[1] < 2:
            st.info(f"Not enough data to train robust models at the series level (rows={X_all.shape[0]}, features={X_all.shape[1]}).")
        else:
            model_choice = st.selectbox("Choose model", ["Elastic Net", "Gradient Boosting", "XGBoost"], index=0)
            test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100.0
            cv_folds  = st.slider("Cross-validation folds", 3, 10, 5, step=1)
            scale_features = st.checkbox("Standardize features (recommended for Elastic Net)", value=(model_choice=="Elastic Net"))

            from sklearn.model_selection import train_test_split, KFold, cross_val_score
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42)

            from sklearn.preprocessing import StandardScaler
            scaler = None
            if scale_features:
                scaler = StandardScaler(with_mean=False)  # sparse-tolerant
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

            model = None
            feat_importance_df = None

            if model_choice == "Elastic Net":
                from sklearn.linear_model import ElasticNetCV
                model = ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,1.0], cv=cv_folds, n_jobs=-1, random_state=42)
            elif model_choice == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
            else:  # XGBoost
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    )
                except Exception as e:
                    st.error(f"XGBoost not available: {e}")
                    model = None

            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                def rmse_compat(y_true, y_pred):
                    try:
                        return mean_squared_error(y_true, y_pred, squared=False)
                    except TypeError:
                        return np.sqrt(mean_squared_error(y_true, y_pred))

                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = rmse_compat(y_test, y_pred)

                c1, c2, c3 = st.columns(3)
                c1.metric("R² (test)", f"{r2:.3f}")
                c2.metric("MAE", f"{mae:.3f}")
                c3.metric("RMSE", f"{rmse:.3f}")

                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
                    st.caption(f"CV R² (train): mean={cv_scores.mean():.3f}, sd={cv_scores.std():.3f}")
                except Exception:
                    pass

                import pandas as pd
                import numpy as np
                if model_choice == "Elastic Net":
                    coef = getattr(model, "coef_", None)
                    if coef is not None:
                        feat_importance_df = (
                            pd.DataFrame({"Feature": X_all.columns, "Effect": coef})
                              .assign(AbsEffect=lambda d: d["Effect"].abs())
                              .sort_values("AbsEffect", ascending=False)
                              .head(20)
                        )
                else:
                    fi = getattr(model, "feature_importances_", None)
                    if fi is not None:
                        feat_importance_df = (
                            pd.DataFrame({"Feature": X_all.columns, "Importance": fi})
                              .sort_values("Importance", ascending=False)
                              .head(20)
                        )

                if feat_importance_df is not None and not feat_importance_df.empty:
                    st.markdown("**Top features (series level)**")
                    st.dataframe(feat_importance_df, use_container_width=True)
                else:
                    st.caption("Model did not expose standard importances/coefficients.")

# ==========================
# 📈 Macro Groups & Subgenres — Year Range + Per-Decade + FacetGrid
# ==========================

# --- NLP Feature Definitions Dictionary ---
feature_definitions = {
    "Anger": "The level of anger present in an act.",
    "Surprise": "The level of surprise present in an act.",
    "Disgust": "The level of disgust present in an act.",
    "Sadness": "The level of sadness present in an act.",
    "Neutral": "The level of neutrality present in an act.",
    "Fear": "The level of fear present in an act.",
    "Joy": "The level of joy present in an act.",
    "Positive": "The amount of positive sentiment present in an act.",
    "Engaged": "High psychological involvement or emotional investment, characterized by increased use of personal pronouns, emotional words, and cognitive processing.",
    "WC": "Total number of words in an act.",
    "Analytic": "The presence of analytical, formal, or logical discussion.",
    "Clout": "The extent of social status, confidence, or leadership discourse.",
    "Authentic": "The presence of honest, non-filtered, or unregulated discussion.",
    "Tone": "A measure of positivity; scores below 50 indicate negativity.",
    "WPS": "The average number of words per sentence in an act.",
    "Sixltr": "Percentage of words with more than six letters.",
    "Dic": "Percentage of words captured in the dictionary.",
    "Cogproc": "Aggregate measure of words indicating active information processing and mental activity, including causation.",
    "Insight": "Words that indicate realizations or understanding.",
    "Cause": "Words that signal causal relationships between elements.",
    "Discrep": "Words that indicate counterfactual thinking (e.g., should, could, would).",
    "Tentat": "Words that indicate uncertainty or possibility (e.g., maybe, perhaps).",
    "Certain": "Words that indicate absolute statements (e.g., always, never).",
    "Differ": "Words that indicate differentiation between concepts (e.g., but, else).",
    "Percept": "Aggregate measure of words describing perception (e.g., look, heard, feel).",
    "See": "Words associated with visual perception.",
    "Hear": "Words associated with auditory perception.",
    "Feel": "Words related to tactile sensation.",
    "Drives": "Aggregate measure of different motivations expressed in an act.",
    "Affiliation": "Words associated with social relationships (e.g., ally, friend).",
    "Achieve": "Words related to success and accomplishment.",
    "Power": "Words related to dominance and hierarchical structures.",
    "Reward": "Words related to receiving benefits or prizes.",
    "Risk": "Words related to danger or uncertainty.",
    "Relativ": "Aggregate measure of words describing spatial relationships (e.g., area, bend, exit).",
    "Motion": "Words related to movement (e.g., arrive, go, car).",
    "Space": "Words indicating spatial direction (e.g., down, in).",
    "Time": "Words related to time duration (e.g., end, until, season)."
}

st.header("Top Features by Year Range — Macro Groups & Subgenres")

# ---------- Config ----------
GENRE_GROUPS = {
    "Comedy & Satire": ["Sitcom", "Comedy-Drama", "Animated sitcom", "Comedy", "Comedy horror"],
    "Drama": ["Drama", "Medical drama", "Family drama", "Period drama", "Political drama", "Psychological drama", "Serial drama"],
    "Crime, Law & Justice": ["Crime drama", "Police procedural", "Legal drama", "Legal thriller", "Crime", "Psychological thriller"],
    "Action & Adventure": ["Action", "Action-adventure", "Superhero", "Adventure", "Action fiction"],
    "Fantasy & Sci-Fi": ["Fantasy", "Science fiction", "Science fantasy", "Supernatural", "Supernatural drama"],
    "Horror & Suspense": ["Horror", "Mystery", "Psychological horror", "Thriller"]
}
TARGET_COL = "Viewership (millions)"
DROP_COLS_BASE = [
    "Show","Cancelled","Season","Episode","Air Date","Air_Date","AirDate","Date","Time",
    "Genre","Aggregate Genre","Year","No.of seasons","Time_in_minutes","Episode Length",
    "No.of episodes","Not engaged_1","Not engaged_2","Not engaged_3", "Decade", TARGET_COL
]
DROP_PREFIXES = ["Network_"]

# ---------- Prep data ----------
_df = data.copy()
if not {"Year","Genre",TARGET_COL}.issubset(_df.columns):
    st.error("Required columns missing: 'Year', 'Genre', and 'Viewership (millions)'.")
    st.stop()

_df["Year"] = pd.to_numeric(_df["Year"], errors="coerce")
_df = _df.dropna(subset=["Year"])
_df["Year"] = _df["Year"].astype(int)
_df = _df[_df["Year"] >= 1950]  # sanity lower bound; adjust if needed
_df["Decade"] = (_df["Year"] // 10) * 10

global_min_year = int(_df["Year"].min())
global_max_year = int(_df["Year"].max())

# --- Safe slider helper (place this once near your other helpers) ---
def safe_year_slider(label, min_y, max_y, *, default_start=None, default_end=None, key=None):
    """
    Returns (yr1, yr2). If min_y == max_y, shows a caption and returns that single year
    without rendering a slider (avoids StreamlitAPIException).
    """
    import pandas as pd

    if pd.isna(min_y) or pd.isna(max_y):
        st.info("No valid years available.")
        return None, None

    min_y, max_y = int(min_y), int(max_y)
    if min_y == max_y:
        st.caption(f"Only one year available for this selection: **{min_y}**")
        return min_y, max_y

    # choose sane defaults if not provided
    start = default_start if default_start is not None else max(min_y, 1980)
    end   = default_end   if default_end   is not None else max_y
    start = max(min_y, min(start, max_y))
    end   = max(start, min(end, max_y))

    return st.slider(
        label,
        min_value=min_y,
        max_value=max_y,
        value=(start, end),
        step=1,
        key=key,
    )

# ---------- Helpers ----------
def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = DROP_COLS_BASE + [c for c in df.columns if any(c.startswith(p) for p in DROP_PREFIXES)]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    return X

def _rf_top_importances(df: pd.DataFrame, top_k: int = 10, scale: bool = True,
                        n_estimators: int = 600, max_depth: int | None = None,
                        random_state: int = 42):
    """
    Returns (feat_df, msg) where feat_df has Feature & Importance (top_k rows).
    msg is None on success or a user-friendly string on failure.
    """
    if TARGET_COL not in df.columns:
        return None, f"Target '{TARGET_COL}' not found."

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    mask = y.notna()
    df = df.loc[mask]
    y = y.loc[mask]

    X = _clean_features(df).loc[y.index]
    X = X.dropna(axis=1, how="all").fillna(0)

    if X.shape[0] < 20:
        return None, f"Not enough rows after filtering (need ≥ 20, have {X.shape[0]})."
    if X.shape[1] < 2:
        return None, f"Not enough valid numeric features (need ≥ 2, have {X.shape[1]})."

    Xn = MinMaxScaler().fit_transform(X) if scale else X.values
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=(None if (max_depth is None or int(max_depth) == 0) else int(max_depth)),
        random_state=int(random_state),
        n_jobs=-1
    )
    rf.fit(Xn, y.values)
    importances = getattr(rf, "feature_importances_", None)
    if importances is None:
        return None, "Model did not expose feature_importances_."

    feat_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return feat_df, None

def _rf_importances_for_year(df: pd.DataFrame, year: int, *, scale: bool,
                             n_estimators: int, max_depth: int | None, random_state: int):
    """Train RF on a single year; return Series of importances indexed by feature (or None if insufficient)."""
    dfy = df[df["Year"] == year]
    if dfy.empty or TARGET_COL not in dfy.columns:
        return None

    y = pd.to_numeric(dfy[TARGET_COL], errors="coerce")
    mask = y.notna()
    if mask.sum() < 15:
        return None

    y = y.loc[mask]
    X = _clean_features(dfy).loc[mask.index]
    X = X.loc[y.index].dropna(axis=1, how="all").fillna(0)
    if X.shape[0] < 15 or X.shape[1] < 2:
        return None

    Xn = MinMaxScaler().fit_transform(X) if scale else X.values
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=(None if (max_depth is None or int(max_depth) == 0) else int(max_depth)),
        random_state=int(random_state),
        n_jobs=-1
    )
    rf.fit(Xn, y.values)
    imps = getattr(rf, "feature_importances_", None)
    if imps is None:
        return None
    return pd.Series(imps, index=X.columns)

def _yearly_importance_df(df: pd.DataFrame, years: list[int], top_feats: list[str], *,
                          scale: bool, n_estimators: int, max_depth: int | None, random_state: int):
    """Return tidy DF with columns: Year, Feature, Importance (and Decade) for requested years/features."""
    rows = []
    for y in sorted(years):
        s = _rf_importances_for_year(df, y, scale=scale, n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=random_state)
        if s is None:
            continue
        for f in top_feats:
            rows.append({"Year": int(y), "Decade": int((y // 10) * 10), "Feature": f, "Importance": float(s.get(f, 0.0))})
    if not rows:
        return pd.DataFrame(columns=["Year", "Decade", "Feature", "Importance"])
    return pd.DataFrame(rows)

def _seaborn_barplot(feat_df: pd.DataFrame, title: str):
    # order features by importance (desc)
    features = (feat_df.sort_values("Importance", ascending=False)["Feature"].tolist())
    # build Blues_r palette mapped to feature names
    cols = sns.color_palette("Blues_r", n_colors=len(features))
    pal_dict = {f: cols[i] for i, f in enumerate(features)}
    pal_list = [pal_dict[f] for f in features]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=feat_df, y="Feature", x="Importance",
        order=features, palette=pal_list, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Random Forest Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)

    # save for the line charts 
    st.session_state["rf_features"] = features
    st.session_state["rf_palette"]  = pal_dict

def _build_line_styling(features, top_k=6):
    """
    Return (palette_dict, dashes_dict, markers_dict, highlight_set)
    for consistent line styling. Uses bar-order from
    st.session_state['rf_features'] when available.
    """

    # Guard: empty list
    if not features:
        return {}, {}, {}, set()

    # Colorblind-safe palette & style cycles
    base_colors = sns.color_palette("tab10", n_colors=min(len(features), 10))
    dash_cycle  = ["solid", "dashed", "dashdot", "dotted"]
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    # Prefer barplot order if present; keep only features that actually exist
    order = st.session_state.get("rf_features", features)
    order = [f for f in order if f in features]

    # Which features get full styling
    k = max(1, min(top_k, len(order)))
    highlight = set(order[:k])

    pal, dashes, markers = {}, {}, {}
    for i, f in enumerate(order):
        if f in highlight:
            pal[f]     = base_colors[i % len(base_colors)]
            dashes[f]  = dash_cycle[i % len(dash_cycle)]
            markers[f] = marker_cycle[i % len(marker_cycle)]
        else:
            pal[f]     = (0.78, 0.78, 0.78)  # light gray
            dashes[f]  = "solid"
            markers[f] = None

    return pal, dashes, markers, highlight

    # colorblind-safe palette + style cycles
    base_colors  = sns.color_palette("tab10", n_colors=min(len(features), 10))
    dash_cycle   = ["solid", "dashed", "dashdot", "dotted"]
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    # prefer the barplot order if it exists
    order = st.session_state.get("rf_features", features)
    order = [f for f in order if f in features]

    highlight = set(order[:min(top_k, len(order))])
    pal, dashes, markers = {}, {}, {}

    for i, f in enumerate(order):
        if f in highlight:
            pal[f]     = base_colors[i % len(base_colors)]
            dashes[f]  = dash_cycle[i % len(dash_cycle)]
            markers[f] = marker_cycle[i % len(marker_cycle)]
        else:
            pal[f]     = (0.78, 0.78, 0.78)  # light gray
            dashes[f]  = "solid"
            markers[f] = None

    return pal, dashes, markers, highlight

def _seaborn_lines(df_long: pd.DataFrame, title: str, top_k_highlight: int = 6):
    if df_long.empty:
        st.info("No per-year data available for the selected slice.")
        return

    # try to reuse the bar order; fall back to uniques in df_long
    bar_features = st.session_state.get("rf_features")
    features_all = list(df_long["Feature"].unique())
    features = [f for f in (bar_features or features_all) if f in features_all]
    present = [f for f in features if f in df_long["Feature"].unique()]

    pal, dashes, markers, highlight = _build_line_styling(features, top_k=top_k_highlight)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # plot each feature manually with matplotlib (linestyle strings are allowed)
    for f in present:
        sub = df_long[df_long["Feature"] == f].sort_values("Year")
        lw = 1.8 if f in highlight else 1.2
        alpha = 1.0 if f in highlight else 0.5
        marker = markers[f] if markers[f] is not None else "o"
        ax.plot(
            sub["Year"], sub["Importance"],
            label=f, color=pal[f], linestyle=dashes[f], marker=marker,
            linewidth=lw, alpha=alpha
        )

    ax.set_title(title)
    ax.set_ylabel("Random Forest Importance")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=.3)

    # legend: only highlighted to cut clutter
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(labels) if lab in highlight]
    if keep:
        ax.legend(
            [handles[i] for i in keep], [labels[i] for i in keep],
            title="Feature (highlighted)", bbox_to_anchor=(1.02, 1), loc="upper left"
        )
    else:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    st.pyplot(fig)

def _facetgrid_lines(df_long: pd.DataFrame, title: str, top_k_highlight: int = 6):
    if df_long.empty:
        st.info("No per-year data available for the selected slice.")
        return

    if "Decade" not in df_long.columns:
        df_long = df_long.assign(Decade=(df_long["Year"] // 10) * 10)
    df_long = df_long.copy()
    df_long["Decade"] = df_long["Decade"].astype(int)

    bar_features = st.session_state.get("rf_features")
    features_all = list(df_long["Feature"].unique())
    features = [f for f in (bar_features or features_all) if f in features_all]
    present = [f for f in features if f in df_long["Feature"].unique()]

    pal, dashes, markers, highlight = _build_line_styling(features, top_k=top_k_highlight)

    # set up facets manually for robustness
    decs = sorted(df_long["Decade"].unique())
    n = len(decs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), sharey=False)
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]

    for ax, dec in zip(axes, decs):
        subd = df_long[df_long["Decade"] == dec]
        for f in present:
            sub = subd[subd["Feature"] == f].sort_values("Year")
            if sub.empty:
                continue
            lw = 1.8 if f in highlight else 1.2
            alpha = 1.0 if f in highlight else 0.5
            marker = markers[f] if markers[f] is not None else "o"
            ax.plot(
                sub["Year"], sub["Importance"],
                label=f, color=pal[f], linestyle=dashes[f], marker=marker,
                linewidth=lw, alpha=alpha
            )
        ax.set_title(f"{dec}s")
        ax.grid(True, alpha=.3)

    # hide any unused axes
    for ax in axes[len(decs):]:
        ax.axis("off")

    # legend (only highlighted), use first axis for handles
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [i for i, lab in enumerate(labels) if lab in highlight]
    if keep:
        fig.legend(
            [handles[i] for i in keep], [labels[i] for i in keep],
            title="Feature (highlighted)", bbox_to_anchor=(0.98, 0.98),
            loc="upper right"
        )
    else:
        fig.legend(bbox_to_anchor=(0.98, 0.98), loc="upper right")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    st.pyplot(fig)

# Definitions lookup
_SYNONYM_MAP = {"Six_Let": "Sixltr", "Dictionary": "Dic"}
def _base_term(name: str) -> str:
    return name.split("_")[0] if "_" in name else name
def _norm_term(t: str) -> str:
    return _SYNONYM_MAP.get(t, t)
def _lookup_definition(term: str) -> str:
    base = _norm_term(_base_term(term))
    return feature_definitions.get(base, "Definition not found in glossary.")

def _decade_starts_in_range(min_y: int, max_y: int) -> list[int]:
    s = (min_y // 10) * 10
    e = (max_y // 10) * 10
    return list(range(s, e + 1, 10))

# ---------- UI (two tabs) ----------
tab_macro, tab_sub = st.tabs(["🏷️ Macro Group", "🔎 Subgenre"])

# ===== Tab 1: Macro Group =====
with tab_macro:
    c0, c1 = st.columns([1.2, 1])
    with c0:
        macro_group = st.selectbox("Macro group", list(GENRE_GROUPS.keys()), key="macro_sel_all")
    with c1:
        view_mode_m = st.radio(
            "View",
            ["Single range", "Per-decade breakdown", "All decades view"],
            horizontal=True,
            key="macro_view_mode_all"
        )

    yr1, yr2 = safe_year_slider(
        "Year range",
        global_min_year,
        global_max_year,
        default_start=max(global_min_year, 1980),
        default_end=global_max_year,
        key="macro_year_range_all2",
    )

    if yr1 is None or yr2 is None:
        st.stop()

    with st.expander("Model settings", expanded=False):
        top_k = st.slider("Top features (K)", 5, 25, 10, step=1, key="macro_topk_all")
        scale_feats = st.checkbox("Scale features (MinMax)", value=True, key="macro_scale_all")
        n_estimators = st.slider("n_estimators", 100, 2000, 600, step=50, key="macro_nest_all")
        max_depth = st.slider("max_depth (0=None)", 0, 40, 0, step=1, key="macro_mdepth_all")
        random_state = st.number_input("random_state", 0, 99999, 42, step=1, key="macro_rs_all")

    mg_df = _df[_df["Genre"].isin(GENRE_GROUPS[macro_group])]
    mg_df = mg_df[(mg_df["Year"] >= yr1) & (mg_df["Year"] <= yr2)]
    st.caption(f"Rows in selection: {len(mg_df)}")

    if view_mode_m == "Single range":
        feat_df, msg = _rf_top_importances(
            mg_df, top_k=top_k, scale=scale_feats,
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        if msg:
            st.info(msg)
        else:
            st.dataframe(feat_df, use_container_width=True)
            _seaborn_barplot(feat_df, f"{macro_group} — Top {top_k} Features ({yr1}–{yr2})")

            years_span = list(range(yr1, yr2 + 1))
            top_feats_macro = feat_df["Feature"].tolist()
            yr_long_macro = _yearly_importance_df(
                mg_df, years_span, top_feats_macro,
                scale=scale_feats, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )
            _seaborn_lines(yr_long_macro, f"{macro_group} — Year-by-Year Importance ({yr1}–{yr2})")

            with st.expander("Definitions for selected features", expanded=False):
                for f in top_feats_macro:
                    st.markdown(f"**{f}** — {_lookup_definition(f)}")

    elif view_mode_m == "Per-decade breakdown":
        decs = _decade_starts_in_range(yr1, yr2)
        present_decs = sorted(set(mg_df["Decade"]).intersection(decs))
        if not present_decs:
            st.info("No decades with data in this range.")
        else:
            tabs = st.tabs([f"{d}s" for d in present_decs])
            for d, t in zip(present_decs, tabs):
                with t:
                    df_dec = mg_df[(mg_df["Decade"] == d)]
                    st.caption(f"Rows: {len(df_dec)}")
                    feat_df, msg = _rf_top_importances(
                        df_dec, top_k=top_k, scale=scale_feats,
                        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
                    )
                    if msg:
                        st.info(msg)
                    else:
                        st.dataframe(feat_df, use_container_width=True)
                        _seaborn_barplot(feat_df, f"{macro_group} — Top {top_k} in the {d}s")

                        years_dec = sorted(df_dec["Year"].unique().tolist())
                        top_feats_dec = feat_df["Feature"].tolist()
                        yr_long_dec = _yearly_importance_df(
                            df_dec, years_dec, top_feats_dec,
                            scale=scale_feats, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
                        )
                        _seaborn_lines(yr_long_dec, f"{macro_group} — Year-by-Year Importance ({d}s)")

                        with st.expander("Definitions for selected features", expanded=False):
                            for f in top_feats_dec:
                                st.markdown(f"**{f}** — {_lookup_definition(f)}")

    else:  # "All decades view" (FacetGrid)
        # Compute top-K on the whole selected range, then plot their yearly trajectories faceted by decade
        feat_df, msg = _rf_top_importances(
            mg_df, top_k=top_k, scale=scale_feats,
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        if msg:
            st.info(msg)
        else:
            st.dataframe(feat_df, use_container_width=True)
            _seaborn_barplot(feat_df, f"{macro_group} — Top {top_k} Features ({yr1}–{yr2})")

            years_span = list(range(yr1, yr2 + 1))
            top_feats_macro = feat_df["Feature"].tolist()
            yr_long_macro = _yearly_importance_df(
                mg_df, years_span, top_feats_macro,
                scale=scale_feats, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )
            _facetgrid_lines(yr_long_macro, f"{macro_group} — Year-by-Year Importance by Decade ({yr1}–{yr2})")

            with st.expander("Definitions for selected features", expanded=False):
                for f in top_feats_macro:
                    st.markdown(f"**{f}** — {_lookup_definition(f)}")

# ===== Tab 2: Subgenre =====
with tab_sub:
    c0, c1, c2 = st.columns([1.2, 1.2, 1])
    with c0:
        macro_for_sub = st.selectbox("Macro group", list(GENRE_GROUPS.keys()), key="sub_macro_sel_all")
        sub_list = GENRE_GROUPS[macro_for_sub]
    with c1:
        subgenre = st.selectbox("Subgenre", sub_list, key="subgenre_sel_all")
    with c2:
        view_mode_s = st.radio(
            "View",
            ["Single range", "Per-decade breakdown", "All decades view"],
            horizontal=True,
            key="sub_view_mode_all"
        )

        yr1s, yr2s = safe_year_slider(
        "Year range (subgenre)",
        global_min_year,
        global_max_year,
        default_start=max(global_min_year, 1980),
        default_end=global_max_year,
        key="sub_year_range_all2",
    )

    if yr1s is None or yr2s is None:
        st.stop()


    with st.expander("Model settings", expanded=False):
        top_k_sg = st.slider("Top features (K)", 5, 25, 10, step=1, key="sub_topk_all")
        scale_feats_sg = st.checkbox("Scale features (MinMax)", value=True, key="sub_scale_all")
        n_estimators_sg = st.slider("n_estimators", 100, 2000, 600, step=50, key="sub_nest_all")
        max_depth_sg = st.slider("max_depth (0=None)", 0, 40, 0, step=1, key="sub_mdepth_all")
        random_state_sg = st.number_input("random_state", 0, 99999, 42, step=1, key="sub_rs_all")

    sg_df = _df[_df["Genre"] == subgenre]
    sg_df = sg_df[(sg_df["Year"] >= yr1s) & (sg_df["Year"] <= yr2s)]
    st.caption(f"Rows in selection: {len(sg_df)}")

    if view_mode_s == "Single range":
        feat_df_sg, msg_sg = _rf_top_importances(
            sg_df, top_k=top_k_sg, scale=scale_feats_sg,
            n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
        )
        if msg_sg:
            st.info(msg_sg)
        else:
            st.dataframe(feat_df_sg, use_container_width=True)
            _seaborn_barplot(feat_df_sg, f"{subgenre} — Top {top_k_sg} Features ({yr1s}–{yr2s})")

            years_span_s = list(range(yr1s, yr2s + 1))
            top_feats_sg = feat_df_sg["Feature"].tolist()
            yr_long_sg = _yearly_importance_df(
                sg_df, years_span_s, top_feats_sg,
                scale=scale_feats_sg, n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
            )
            _seaborn_lines(yr_long_sg, f"{subgenre} — Year-by-Year Importance ({yr1s}–{yr2s})")

            with st.expander("Definitions for selected features", expanded=False):
                for f in top_feats_sg:
                    st.markdown(f"**{f}** — {_lookup_definition(f)}")

    elif view_mode_s == "Per-decade breakdown":
        decs_s = _decade_starts_in_range(yr1s, yr2s)
        present_decs_s = sorted(set(sg_df["Decade"]).intersection(decs_s))
        if not present_decs_s:
            st.info("No decades with data in this range.")
        else:
            tabs_s = st.tabs([f"{d}s" for d in present_decs_s])
            for d, t in zip(present_decs_s, tabs_s):
                with t:
                    df_dec = sg_df[(sg_df["Decade"] == d)]
                    st.caption(f"Rows: {len(df_dec)}")
                    feat_df_sg, msg_sg = _rf_top_importances(
                        df_dec, top_k=top_k_sg, scale=scale_feats_sg,
                        n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
                    )
                    if msg_sg:
                        st.info(msg_sg)
                    else:
                        st.dataframe(feat_df_sg, use_container_width=True)
                        _seaborn_barplot(feat_df_sg, f"{subgenre} — Top {top_k_sg} in the {d}s")

                        years_dec_s = sorted(df_dec["Year"].unique().tolist())
                        top_feats_dec_s = feat_df_sg["Feature"].tolist()
                        yr_long_dec_s = _yearly_importance_df(
                            df_dec, years_dec_s, top_feats_dec_s,
                            scale=scale_feats_sg, n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
                        )
                        _seaborn_lines(yr_long_dec_s, f"{subgenre} — Year-by-Year Importance ({d}s)")

                        with st.expander("Definitions for selected features", expanded=False):
                            for f in top_feats_dec_s:
                                st.markdown(f"**{f}** — {_lookup_definition(f)}")

    else:  # "All decades view" (FacetGrid)
        feat_df_sg, msg_sg = _rf_top_importances(
            sg_df, top_k=top_k_sg, scale=scale_feats_sg,
            n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
        )
        if msg_sg:
            st.info(msg_sg)
        else:
            st.dataframe(feat_df_sg, use_container_width=True)
            _seaborn_barplot(feat_df_sg, f"{subgenre} — Top {top_k_sg} Features ({yr1s}–{yr2s})")

            years_span_s = list(range(yr1s, yr2s + 1))
            top_feats_sg = feat_df_sg["Feature"].tolist()
            yr_long_sg = _yearly_importance_df(
                sg_df, years_span_s, top_feats_sg,
                scale=scale_feats_sg, n_estimators=n_estimators_sg, max_depth=max_depth_sg, random_state=random_state_sg
            )
            _facetgrid_lines(yr_long_sg, f"{subgenre} — Year-by-Year Importance by Decade ({yr1s}–{yr2s})")

            with st.expander("Definitions for selected features", expanded=False):
                for f in top_feats_sg:
                    st.markdown(f"**{f}** — {_lookup_definition(f)}")

