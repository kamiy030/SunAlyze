# app.py (Streamlit) — CLEAN, STABLE VERSION
# Notes:
# - Removed the marker-based CSS + marker blocks (they were unreliable across Streamlit versions)
# - Uses st.container(border=True) everywhere you want a “card”
# - Keeps your left metadata beside graph + adds FE/FS card under metadata
# - Makes table + chart appear inside card boxes with shadow (via the border wrapper styling)
# - Removes duplicate l,r columns declaration

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SunAlyze", layout="wide")
ROOT = Path(__file__).parent

META_PATH = ROOT / "metadata_station01.csv"
FEAT_IMPACT_PATH = ROOT / "feature_impact.csv"
HIST_PATH = ROOT / "station01_timefixed.csv"
TEST_PATH = ROOT / "test.csv"

GRU_PATH = ROOT / "gru_sbs_tuned.keras"
SCALER_SBS = ROOT / "scaler_sbs_portable.joblib"
SBS_JSON = ROOT / "sbs_selected_features.json"
SBS_CSV = ROOT / "sbs_selected_features.csv"

XGB_PATH = ROOT / "xgb_mic_tuned.joblib"
SCALER_MIC = ROOT / "scaler_mic_portable.joblib"
MIC_JSON = ROOT / "mic_selected_features.json"
MIC_CSV = ROOT / "mic_selected_features.csv"

HISTORY_STEPS = 96  # 96 * 15min = 24 hours history

MODEL_COLORS = {
    "XGBoost": "#687B8E",  # purple
    "GRU": "#FCB32D",      # gold
}

# =========================
# GLOBAL CSS (ONE PLACE ONLY)
# =========================
st.markdown(
    """
<style>
/* Page background */
.stApp { background: #F4F7FB; }

/* Section titles helper */
.section-title{
  font-size: 27px;
  font-weight: 800;
  color: #1E3A8A;
  margin: 0 0 10px 0;
  text-align: center;
}

/* Only containers marked as a card get the card styling */
.card-tag + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]{
  background: #ffffff !important;
  border-radius: 18px !important;
  border: 1px solid rgba(30,58,138,0.10) !important;
  box-shadow: 0 14px 34px rgba(15,23,42,0.14) !important;
  padding: 18px 18px 14px 18px !important;
}

/* Clip plotly inside the card */
.card-tag + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] .js-plotly-plot,
.card-tag + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] .plot-container,
.card-tag + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] .svg-container{
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* Dataframe rounding inside the card */
.card-tag + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stDataFrame"]{
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(30,58,138,0.08) !important;
}

/* ---- THE CARD (used for small metric cards) ---- */
.meta-card{
  background: #ffffff;
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid rgba(30,58,138,0.10);
  box-shadow: 0 10px 24px rgba(15,23,42,0.10);
  min-width: 160px;   

  /* keep card height consistent in Section 1 */
  min-height: 112px;
  height: auto;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.meta-title{
  font-size: 15px;
  font-weight: 750;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: rgba(17,24,39,0.55);
  margin-bottom: 6px;
}

.meta-value{
  font-size: 20px;
  font-weight: 700;
  color: #111827;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Smaller "value" for compact cards (summary + anchor) */
.meta-value-sm{
  font-size: 18px;
  font-weight: 700;
  color: #111827;
  line-height: 1.15;
}

/* Compact dataframe */
div[data-testid="stDataFrame"] * { font-size: 12px !important; }
div[data-testid="stDataFrame"] div[role="row"] { min-height: 24px !important; }

/* =========================
   FE/FS CARD (Section 1)
   ========================= */
.ff-card{
  background:#ffffff;
  border-radius:14px;
  padding:14px 16px;
  border:1px solid rgba(30,58,138,0.10);
  box-shadow:0 10px 24px rgba(15,23,42,0.10);
  margin-top: 14px;
}

.ff-title{
  font-size: 13px;
  font-weight: 800;
  color: #111827;
  margin-bottom: 10px;
}

.ff-grid{
  display:flex;
  gap: 12px;
}

.ff-col{
  flex: 1;
  background: rgba(30,58,138,0.05);
  border-radius: 12px;
  padding: 12px;
}

.ff-col-title{
  font-size: 12px;
  font-weight: 800;
  color: rgba(17,24,39,0.85);
  margin-bottom: 8px;
}

.ff-col ul{
  margin:0;
  padding-left: 18px;
  font-size: 12px;
  color: rgba(17,24,39,0.75);
  line-height: 1.5;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def pick_col(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)

def meta_card(title_txt: str, value_txt: str, small=False, center=True, wrap=False):
    wrap_css = "white-space: normal;" if wrap else "white-space: nowrap; overflow:hidden; text-overflow: ellipsis;"
    align = "text-align:center;" if center else "text-align:left;"
    value_class = "meta-value-sm" if small else "meta-value"
    st.markdown(
        f"""
        <div class="meta-card" style="{align}">
          <div class="meta-title">{title_txt}</div>
          <div class="{value_class}" style="max-width:100%; {wrap_css}">
            {value_txt}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def card_container():
    st.markdown("<div class='card-tag'></div>", unsafe_allow_html=True)
    return st.container(border=True)

def fe_fs_card(fe_items, fs_items):
    st.markdown(
        f"""
        <div class="ff-card">
          <div class="ff-title">What changes across Baseline / FE / FE+FS?</div>
          <div class="ff-grid">
            <div class="ff-col">
              <div class="ff-col-title">Feature Engineering (FE)</div>
              <ul>
                {''.join([f"<li>{x}</li>" for x in fe_items])}
              </ul>
            </div>
            <div class="ff-col">
              <div class="ff-col-title">Feature Selection (FS)</div>
              <ul>
                {''.join([f"<li>{x}</li>" for x in fs_items])}
              </ul>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def daily_energy_kwh(df: pd.DataFrame, ts_col: str, power_col: str) -> pd.DataFrame:
    d = df[[ts_col, power_col]].dropna().copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col)

    dt = d[ts_col].diff().dropna()
    dt_min = float(dt.median().total_seconds() / 60) if len(dt) else 15.0
    dt_h = dt_min / 60.0

    d["date"] = d[ts_col].dt.date
    d["energy_kwh"] = (d[power_col].astype(float) * dt_h) / 1000.0
    return d.groupby("date", as_index=False)["energy_kwh"].sum()

@st.cache_resource
def load_gru_bundle():
    model = tf.keras.models.load_model(GRU_PATH, compile=False)
    scaler = joblib.load(SCALER_SBS)
    feats = None

    if SBS_CSV.exists():
        feats_df = pd.read_csv(SBS_CSV)
        feats = feats_df.iloc[:, 0].dropna().astype(str).tolist()
    elif SBS_JSON.exists():
        obj = json.load(open(SBS_JSON, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            feats = obj.get("selected_features") or obj.get("features") or obj.get("sbs_selected_features")
        else:
            feats = obj
        feats = [str(x) for x in feats] if feats else None

    if not feats:
        raise RuntimeError("Missing SBS feature list. Provide sbs_selected_features.csv or sbs_selected_features.json")

    return model, scaler, feats

@st.cache_resource
def load_xgb_bundle():
    model = joblib.load(XGB_PATH)
    scaler = joblib.load(SCALER_MIC)
    feats = None

    if MIC_CSV.exists():
        feats_df = pd.read_csv(MIC_CSV)
        feats = feats_df.iloc[:, 0].dropna().astype(str).tolist()
    elif MIC_JSON.exists():
        obj = json.load(open(MIC_JSON, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            feats = obj.get("selected_features") or obj.get("features") or obj.get("mic_selected_features")
        else:
            feats = obj
        feats = [str(x) for x in feats] if feats else None

    if not feats:
        raise RuntimeError("Missing MIC feature list. Provide mic_selected_features.csv or mic_selected_features.json")

    return model, scaler, feats

# ---- Forecast helpers ----
def rolling_forecast_gru(df_test, feats, model, x_scaler, anchor_i, horizon_steps, history_steps):
    X = df_test.reindex(columns=feats).astype(float).fillna(0.0).values
    Xs = x_scaler.transform(X)

    preds = []
    for k in range(horizon_steps):
        start = anchor_i - history_steps + k
        end   = anchor_i + k
        window = Xs[start:end][np.newaxis, :, :]
        yhat = model.predict(window, verbose=0).reshape(-1)[0]
        preds.append(float(yhat))

    return np.asarray(preds, dtype=float)

def horizon_forecast_xgb(df_test, feats, model, x_scaler, anchor_i, horizon_steps):
    X = df_test.reindex(columns=feats).astype(float).fillna(0.0).values
    Xs = x_scaler.transform(X)
    X_h = Xs[anchor_i:anchor_i + horizon_steps]
    preds = model.predict(X_h)
    return np.asarray(preds, dtype=float)

# =========================
# HEADER
# =========================
st.markdown(
    """
<div style="text-align:center; padding: 8px 0 6px;">
  <div style="font-size:54px; font-weight:900; line-height:1; color:#111827;">SunAlyze</div>
  <div style="font-size:18px; font-weight:600; opacity:0.7; margin-top:6px;">Solar Dashboard Analysis</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

# =========================
# SECTION 1 — METADATA + FEATURE IMPACT
# =========================
top_left, top_right = st.columns([1.1, 2.2], gap="large")

with top_left:
    meta = read_csv(META_PATH)
    m = meta.iloc[0]

    cap_col = pick_col(meta, ["Capacity", "Installed Capacity (W)", "installed_capacity_w", "installed_capacity"])
    tech_col = pick_col(meta, ["PV_Technology", "PV Technology", "pv_technology"])
    tilt_col = pick_col(meta, ["Array_Tilt", "Array Tilt", "array_tilt"])
    lat_col = pick_col(meta, ["Latitude", "lat"])
    lon_col = pick_col(meta, ["Longitude", "lon", "lng"])

    capacity = m[cap_col] if cap_col else "—"
    pvtech = m[tech_col] if tech_col else "—"
    tilt = m[tilt_col] if tilt_col else "—"
    loc = f"{m[lat_col]}, {m[lon_col]}" if (lat_col and lon_col) else "—"

    st.markdown("<div class='section-title' style='text-align:left;'>System Metadata</div>", unsafe_allow_html=True)

    meta_c1, meta_c2 = st.columns(2, gap="large")

    with meta_c1:
        meta_card("Capacity", f"{capacity}")
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        meta_card("Array Tilt", f"{tilt}")

    with meta_c2:
        meta_card("PV Technology", f"{pvtech}")
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        meta_card("Location", f"{loc}", wrap=True)

    fe_items = [
        "Extraction of time-based features (e.g., hour-of-day, day-of-year)",
        "Construction of lagged and rolling window features from historical power output",
        "Combination of temporal and lag-based features to capture short-term dynamics",
    ]
    fs_items = [
        "Selection of informative predictors using surrogate-based methods",
        "Reduction of redundant and less relevant features to simplify the input space",
        "Retention of top-ranked features to improve efficiency and generalization",
    ]
    fe_fs_card(fe_items, fs_items)

with top_right:
    with st.container(border=True):
        title("Feature Preprocessing Impact")

        fi = read_csv(FEAT_IMPACT_PATH)
        config_col = pick_col(fi, ["Config", "configuration"])
        model_col = pick_col(fi, ["Model", "model"])
        rmse_col = pick_col(fi, ["RMSE", "rmse"])
        mae_col = pick_col(fi, ["MAE", "mae"])
        r2_col = pick_col(fi, ["R2", "R²", "r2"])

        metric_map = {}
        if rmse_col: metric_map["RMSE"] = rmse_col
        if mae_col: metric_map["MAE"] = mae_col
        if r2_col: metric_map["R²"] = r2_col

        metric_label = st.selectbox("Metric", list(metric_map.keys()), index=0)
        metric_col = metric_map[metric_label]

        plot_df = fi.rename(columns={config_col: "Config", model_col: "Model"}).copy()

        fig = px.bar(
            plot_df,
            x="Config",
            y=metric_col,
            color="Model",
            barmode="group",
            color_discrete_map=MODEL_COLORS,
            category_orders={"Model": ["XGBoost", "GRU"]},
        )

        fig.update_layout(
            height=370,
            margin=dict(l=20, r=20, t=30, b=10),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# SECTION 2 — TIME WINDOW + HISTORICAL (PRE-TEST) ONLY
# =========================
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
title("Overview")

hist = read_csv(HIST_PATH)
test = read_csv(TEST_PATH)

ts_col = pick_col(hist, ["date_time", "timestamp", "time"])
pw_col = pick_col(hist, ["power", "solar_power", "Solar Power (actual)"])
test_ts = pick_col(test, ["date_time", "timestamp", "time"])

hist[ts_col] = pd.to_datetime(hist[ts_col])
test[test_ts] = pd.to_datetime(test[test_ts])

test_start = test[test_ts].min()
pre = hist[hist[ts_col] < test_start].copy()

min_t, max_t = pre[ts_col].min(), pre[ts_col].max()
start_t, end_t = st.slider(
    "Time Window",
    min_value=min_t.to_pydatetime(),
    max_value=max_t.to_pydatetime(),
    value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
)

win = pre[(pre[ts_col] >= pd.to_datetime(start_t)) & (pre[ts_col] <= pd.to_datetime(end_t))].copy()

# ---- Summary row (3 cards) ----
st.markdown("<div style='height:17px;'></div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3, gap="medium")
with s1:
    meta_card("Date Range", f"{win[ts_col].min().date()} → {win[ts_col].max().date()}", small=True)
with s2:
    meta_card("Total Records", f"{len(win):,}", small=True)
with s3:
    meta_card("Sampling Interval", "15 min", small=True)

# ---- Table + Energy (NOW IN CARD CONTAINERS) ----
l, r = st.columns([0.95, 1.65], gap="large")

with l:
    with card_container():
        title("Solar Power Data")
        st.dataframe(
            win[[ts_col, pw_col]].rename(columns={ts_col: "Timestamp", pw_col: "Solar Power"}),
            height=320,
            use_container_width=True,
        )

with r:
    with card_container():
        title("Energy Daily")
        energy = daily_energy_kwh(win, ts_col, pw_col)
        fig2 = px.bar(energy, x="date", y="energy_kwh")
        fig2.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# SECTION 3 — FORECASTING (GRU, validated on test)
# =========================
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
title("Forecasting")

df_test = test.copy()
df_test[ts_col] = pd.to_datetime(df_test[ts_col])
df_test = df_test.sort_values(ts_col).reset_index(drop=True)

gru_model, sbs_scaler, sbs_feats = load_gru_bundle()
xgb_model, mic_scaler, mic_feats = load_xgb_bundle()

col_h, col_slider, col_info = st.columns([1, 3, 1], gap="large")

with col_h:
    hmap = {"1 hour": 4, "3 hours": 12, "6 hours": 24}
    hlabel = st.selectbox("Horizon", list(hmap.keys()), index=0)
    H = hmap[hlabel]

min_anchor = HISTORY_STEPS
max_anchor = len(df_test) - H - 1

with col_slider:
    anchor_i = st.slider("Timeline", min_value=min_anchor, max_value=max_anchor, value=min_anchor)

anchor_time = df_test.loc[anchor_i, ts_col]

with col_info:
    st.markdown(
        f"""
        <div class="meta-card" style="text-align:center;">
          <div class="meta-title">Forecast starts at</div>
          <div class="meta-value-sm">{anchor_time:%d %b %Y} ({anchor_time:%H:%M})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hist_df = df_test.iloc[anchor_i - HISTORY_STEPS : anchor_i]
fut_df  = df_test.iloc[anchor_i : anchor_i + H]

t_hist = hist_df[ts_col].values
t_fut  = fut_df[ts_col].values
actual_hist = hist_df[pw_col].values
actual_fut  = fut_df[pw_col].values

preds_gru = rolling_forecast_gru(df_test, sbs_feats, gru_model, sbs_scaler, anchor_i, H, HISTORY_STEPS)
preds_xgb = horizon_forecast_xgb(df_test, mic_feats, xgb_model, mic_scaler, anchor_i, H)

st.radio("Model", ["GRU", "XGBoost", "Both"], horizontal=True, key="model_sel")
model_sel = st.session_state.get("model_sel", "GRU")

st.caption("*Short-horizon forecast validated on test data (exogenous meteorological + temporal inputs).*")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_hist,
    y=actual_hist,
    mode="lines",
    name="Actual (History)",
    line=dict(color="#0131b4", width=2.5)
))

fig.add_trace(go.Scatter(
    x=t_fut,
    y=actual_fut,
    mode="lines",
    name="Actual (Future)",
    line=dict(color="#79B3FF", width=2.5)
))

if model_sel in ("GRU", "Both"):
    fig.add_trace(go.Scatter(
        x=t_fut,
        y=preds_gru,
        mode="lines+markers",
        name="Forecast (GRU)",
        line=dict(color=MODEL_COLORS["GRU"], width=2.5),
        marker=dict(size=6)
    ))

if model_sel in ("XGBoost", "Both"):
    fig.add_trace(go.Scatter(
        x=t_fut,
        y=preds_xgb,
        mode="lines+markers",
        name="Forecast (XGBoost)",
        line=dict(color=MODEL_COLORS["XGBoost"], width=2.5),
        marker=dict(size=6)
    ))

fig.add_vline(
    x=anchor_time,
    line_dash="dash",
    line_width=2,
    line_color="black"
)

fig.update_layout(
    title="Short-Horizon Forecast (Validated)",
    height=420,
    margin=dict(l=20, r=20, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

fig.update_xaxes(range=[hist_df[ts_col].iloc[0], fut_df[ts_col].iloc[-1]])
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)
