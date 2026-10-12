from pathlib import Path
import os
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Delhi-NCR AQI Intelligence", page_icon="🌫️", layout="wide")

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "o3"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_WINDOWS = [6, 24, 168]
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-01-01")
CITY_COLORS = ["#0ea5e9", "#8b5cf6", "#14b8a6", "#f59e0b", "#ef4444"]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 14px 16px; border-radius: 12px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#0f172a,#164e63); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin-bottom:.25rem;}
.hero p {color:#dbeafe; margin-bottom:0;}
.small-note {font-size:.85rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def download_kaggle_dataset():
    import kagglehub
    path = kagglehub.dataset_download("aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024")
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("Kaggle download completed but no CSV file was found.")
    # Prefer a CSV whose name suggests processed/hourly air-quality data.
    preferred = [p for p in csvs if "processed" in os.path.basename(p).lower()]
    selected = preferred[0] if preferred else csvs[0]
    return pd.read_csv(selected), selected

@st.cache_data(show_spinner=False)
def load_uploaded_bytes(file_bytes, filename):
    import io, zipfile
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise FileNotFoundError("The uploaded ZIP contains no CSV files.")
            chosen = next((n for n in names if "processed" in os.path.basename(n).lower()), names[0])
            with zf.open(chosen) as f:
                return pd.read_csv(f), chosen
    return pd.read_csv(io.BytesIO(file_bytes)), filename

def normalize_data(raw):
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    # Allow common timestamp aliases.
    if "timestamp" not in d.columns:
        for alias in ["datetime", "date_time", "date"]:
            if alias in d.columns:
                d = d.rename(columns={alias: "timestamp"})
                break
    # Allow common city/location aliases.
    if "location_name" not in d.columns:
        for alias in ["city", "location", "station", "locationname"]:
            if alias in d.columns:
                d = d.rename(columns={alias: "location_name"})
                break
    # Support common pollutant spellings.
    aliases = {"pm2.5": "pm25", "pm_2_5": "pm25", "pm2_5": "pm25",
               "pm10.0": "pm10", "nitrogen_dioxide": "no2",
               "sulfur_dioxide": "so2", "sulphur_dioxide": "so2",
               "ozone": "o3", "carbon_monoxide": "co"}
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d.columns})
    required = ["timestamp", "location_name", "aqi"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {d.columns.tolist()}")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["location_name"] = d["location_name"].astype(str)
    for c in ["aqi"] + [p for p in POLLUTANTS if p in d.columns]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "location_name", "aqi"]).copy()
    for p in POLLUTANTS:
        if p not in d.columns:
            d[p] = np.nan
    d = d.sort_values(["location_name", "timestamp"]).drop_duplicates(
        ["location_name", "timestamp"], keep="last"
    ).reset_index(drop=True)
    d["year"] = d["timestamp"].dt.year
    d["month"] = d["timestamp"].dt.month
    d["hour"] = d["timestamp"].dt.hour
    d["day_of_week"] = d["timestamp"].dt.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["date"] = d["timestamp"].dt.floor("D")
    return d

def category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

def build_supervised(frame, horizon):
    d = frame.sort_values(["location_name", "timestamp"]).reset_index(drop=True).copy()
    g = d.groupby("location_name")["aqi"]
    feats = {f"aqi_lag_{k}": g.shift(k) for k in LAGS}
    for w in ROLL_WINDOWS:
        feats[f"aqi_roll_mean_{w}"] = g.transform(lambda s, w=w: s.rolling(w).mean())
    feats["aqi_roll_std_24"] = g.transform(lambda s: s.rolling(24).std())
    feats["all_city_mean_now"] = d.groupby("timestamp")["aqi"].transform("mean")
    target_time = d["timestamp"] + pd.Timedelta(hours=horizon)
    hour, doy = target_time.dt.hour, target_time.dt.dayofyear
    calendar = pd.DataFrame({
        "target_hour_sin": np.sin(2*np.pi*hour/24),
        "target_hour_cos": np.cos(2*np.pi*hour/24),
        "target_doy_sin": np.sin(2*np.pi*doy/365.25),
        "target_doy_cos": np.cos(2*np.pi*doy/365.25),
    })
    cities = sorted(d["location_name"].unique())
    city_dummies = pd.DataFrame({f"city_{c}": (d["location_name"] == c).astype(int) for c in cities})
    X = pd.concat([d[["aqi"] + POLLUTANTS].reset_index(drop=True),
                   pd.DataFrame(feats).reset_index(drop=True),
                   calendar.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)
    y = g.shift(-horizon)
    meta = pd.DataFrame({"timestamp": d["timestamp"], "location_name": d["location_name"],
                         "target_time": target_time})
    keep = X.notna().all(axis=1) & y.notna()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True), meta.loc[keep].reset_index(drop=True)

def split_masks(meta):
    t = meta["target_time"]
    return (t < TRAIN_END).to_numpy(), ((t >= TRAIN_END) & (t < VAL_END)).to_numpy(), ((t >= VAL_END) & (t < pd.Timestamp("2024-01-01"))).to_numpy()

def climatology(meta, y, train_mask):
    z = pd.DataFrame({"month": meta["target_time"].dt.month, "hour": meta["target_time"].dt.hour, "y": y})
    table = z.loc[train_mask].groupby(["month", "hour"])["y"].mean().rename("clim").reset_index()
    return z[["month", "hour"]].merge(table, on=["month", "hour"], how="left")["clim"].fillna(y[train_mask].mean()).to_numpy()

@st.cache_resource(show_spinner="Training forecasting models on the dataset...")
def train_models(data_fingerprint, frame):
    trained = {}
    metrics = []
    for h in [1, 24]:
        X, y, meta = build_supervised(frame, h)
        tr, va, te = split_masks(meta)
        if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
            continue
        preds = {
            "Persistence": X.loc[te, "aqi"].to_numpy(),
            "Climatology (month × hour)": climatology(meta, y, tr)[te]
        }
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        ridge.fit(X.loc[tr], y.loc[tr])
        preds["Ridge"] = ridge.predict(X.loc[te])
        xgb = XGBRegressor(
            n_estimators=450, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X.loc[tr], y.loc[tr], eval_set=[(X.loc[va], y.loc[va])], verbose=False)
        preds["XGBoost"] = xgb.predict(X.loc[te])
        for name, pred in preds.items():
            metrics.append({"Horizon": f"{h} hour" if h == 1 else "24 hours",
                            "Model": name,
                            "MAE": mean_absolute_error(y.loc[te], pred),
                            "RMSE": np.sqrt(mean_squared_error(y.loc[te], pred)),
                            "R²": r2_score(y.loc[te], pred)})
        trained[h] = {"model": xgb, "ridge": ridge, "X": X, "y": y, "meta": meta,
                      "train": tr, "val": va, "test": te, "preds": preds,
                      "test_y": y.loc[te].to_numpy(), "test_meta": meta.loc[te].reset_index(drop=True)}
    return trained, pd.DataFrame(metrics)

st.markdown("""
<div class="hero">
  <h1>🌫️ Delhi-NCR AQI Intelligence</h1>
  <p>Explore historical air quality, compare cities, inspect pollutant patterns, and evaluate short-term AQI forecasts.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data source", ["Download from Kaggle", "Upload CSV / ZIP"], index=0)
    uploaded = st.file_uploader("Upload dataset", type=["csv", "zip"]) if source == "Upload CSV / ZIP" else None
    st.caption("Expected columns include timestamp, location_name, aqi, and pollutant fields such as pm25 and pm10.")
    st.divider()
    st.caption("Model horizons: 1 hour and 24 hours. Test period: calendar year 2023.")

try:
    if source == "Download from Kaggle":
        raw, source_name = download_kaggle_dataset()
    else:
        if uploaded is None:
            st.info("Upload your dataset CSV or ZIP in the sidebar to begin.")
            st.stop()
        raw, source_name = load_uploaded_bytes(uploaded.getvalue(), uploaded.name)
    data = normalize_data(raw)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info("For Kaggle download, install `kagglehub`. If Kaggle access requires credentials, configure them in your environment or upload the CSV/ZIP instead.")
    st.stop()

if data.empty:
    st.error("The loaded dataset has no valid rows after parsing timestamp, location and AQI.")
    st.stop()

cities = sorted(data["location_name"].unique().tolist())
with st.sidebar:
    st.divider()
    selected_cities = st.multiselect("Cities", cities, default=cities)
    date_min, date_max = data["timestamp"].min().date(), data["timestamp"].max().date()
    date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    show_all = st.checkbox("Show full data table", value=False)

if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
else:
    start_date, end_date = pd.Timestamp(date_min), pd.Timestamp(date_max) + pd.Timedelta(days=1)
filtered = data[data["location_name"].isin(selected_cities) &
                data["timestamp"].between(start_date, end_date)].copy()
if filtered.empty:
    st.warning("No records match the selected cities and dates.")
    st.stop()

latest_time = data["timestamp"].max()
latest = data[data["timestamp"] == latest_time]
avg_latest = latest["aqi"].mean()
mean_aqi = filtered["aqi"].mean()
peak_row = filtered.loc[filtered["aqi"].idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average AQI", f"{mean_aqi:.1f}", help="Mean across the selected cities and date range.")
m2.metric("Latest dataset AQI", f"{avg_latest:.1f}", category(avg_latest) if pd.notna(avg_latest) else "Unavailable")
m3.metric("Peak AQI in selection", f"{peak_row['aqi']:.1f}")
m4.metric("Records in selection", f"{len(filtered):,}")

if latest_time < pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=2):
    st.warning(f"The latest timestamp in this dataset is **{latest_time}**. Forecasts are historical model outputs based on the dataset, not live AQI readings.")

tab_overview, tab_trends, tab_pollutants, tab_forecast, tab_quality = st.tabs(
    ["Overview", "Trends & cities", "Pollutants", "Forecast & model evaluation", "Data quality"]
)

with tab_overview:
    left, right = st.columns([1.5, 1])
    with left:
        daily = filtered.groupby(filtered["timestamp"].dt.floor("D"), as_index=False)["aqi"].mean()
        daily.columns = ["Date", "Average AQI"]
        fig = px.line(daily, x="Date", y="Average AQI", title="AQI over time (daily average)",
                      template="plotly_white")
        fig.update_traces(line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cats = filtered["aqi"].map(category).value_counts().rename_axis("Category").reset_index(name="Records")
        fig = px.pie(cats, names="Category", values="Records", hole=0.55,
                     title="AQI category mix", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Snapshot by city")
    city_summary = filtered.groupby("location_name").agg(
        Mean_AQI=("aqi", "mean"), Median_AQI=("aqi", "median"),
        Maximum_AQI=("aqi", "max"), Records=("aqi", "size")
    ).sort_values("Mean_AQI", ascending=False).reset_index()
    st.dataframe(city_summary.round(2), use_container_width=True, hide_index=True)

with tab_trends:
    city_series = filtered.groupby([filtered["timestamp"].dt.floor("D"), "location_name"], as_index=False)["aqi"].mean()
    city_series.columns = ["Date", "City", "AQI"]
    fig = px.line(city_series, x="Date", y="AQI", color="City", title="Daily AQI by city",
                  template="plotly_white", color_discrete_sequence=CITY_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        month_avg = filtered.groupby(filtered["timestamp"].dt.month)["aqi"].mean().reindex(range(1, 13))
        month_avg.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = px.bar(month_avg.reset_index(), x="index", y="aqi", labels={"index":"Month","aqi":"Mean AQI"},
                     title="Seasonality: mean AQI by month", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hour_avg = filtered.groupby(filtered["timestamp"].dt.hour)["aqi"].mean().reindex(range(24))
        fig = px.line(x=hour_avg.index, y=hour_avg.values, labels={"x":"Hour of day","y":"Mean AQI"},
                      title="Average AQI by hour", markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab_pollutants:
    available_pollutants = [p for p in POLLUTANTS if p in filtered.columns and filtered[p].notna().any()]
    chosen_pollutant = st.selectbox("Choose pollutant", available_pollutants, index=available_pollutants.index("pm25") if "pm25" in available_pollutants else 0)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x=chosen_pollutant, nbins=50, title=f"{chosen_pollutant.upper()} distribution",
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(filtered.sample(min(5000, len(filtered)), random_state=42),
                         x=chosen_pollutant, y="aqi", color="location_name",
                         title=f"{chosen_pollutant.upper()} vs AQI (sampled)", template="plotly_white",
                         opacity=0.45)
        st.plotly_chart(fig, use_container_width=True)
    corr_cols = [p for p in available_pollutants if filtered[p].notna().sum() > 2] + ["aqi"]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Pollutant / AQI correlation matrix",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Correlation describes association, not causation. Same-time pollutant measurements may reconstruct AQI rather than provide a genuine future forecast.")

with tab_forecast:
    st.subheader("Model evaluation on the held-out 2023 period")
    st.write("The app trains Ridge and XGBoost for 1-hour and 24-hour AQI forecasts and compares them with persistence and month-hour climatology baselines.")
    if not st.button("Train / load forecasting models", type="primary"):
        st.info("Click the button to train the models. Results are cached for this session.")
    else:
        try:
            fingerprint = f"{len(data)}-{data['timestamp'].min()}-{data['timestamp'].max()}-{data['aqi'].sum():.3f}"
            model_data = data.dropna(subset=["aqi"]).copy()
            models, metric_table = train_models(fingerprint, model_data)
            if metric_table.empty:
                st.warning("Could not create train/validation/test splits. Check whether the data spans 2019–2023.")
            else:
                st.dataframe(metric_table.round(3), use_container_width=True, hide_index=True)
                fig = px.bar(metric_table, x="Horizon", y="MAE", color="Model", barmode="group",
                             title="Held-out MAE by horizon (lower is better)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                horizon_choice = st.selectbox("Inspect test predictions", ["1 hour", "24 hours"])
                h = 1 if horizon_choice == "1 hour" else 24
                if h in models:
                    bundle = models[h]
                    test_y = bundle["test_y"]
                    pred = bundle["preds"]["XGBoost"]
                    meta_test = bundle["test_meta"]
                    city_filter = st.selectbox("City for prediction plot", sorted(meta_test["location_name"].unique()))
                    mask = meta_test["location_name"].eq(city_filter).to_numpy()
                    plot_df = pd.DataFrame({"Target time": meta_test.loc[mask, "target_time"],
                                            "Actual AQI": test_y[mask], "XGBoost prediction": pred[mask]})
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_df["Target time"], y=plot_df["Actual AQI"], name="Actual", mode="lines"))
                    fig.add_trace(go.Scatter(x=plot_df["Target time"], y=plot_df["XGBoost prediction"], name="XGBoost", mode="lines"))
                    fig.update_layout(title=f"{city_filter}: actual vs predicted ({h}h ahead)", template="plotly_white",
                                      xaxis_title="Target time", yaxis_title="AQI")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("This is a retrospective test-year evaluation, not a live forecast. The dataset ends in early 2024.")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.exception(e)

with tab_quality:
    st.subheader("Dataset audit")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(data):,}")
    q2.metric("Columns", len(data.columns))
    q3.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    q4.metric("Duplicate city-time pairs", f"{int(data.duplicated(['timestamp','location_name']).sum()):,}")
    st.write("**Time coverage**")
    st.write(f"{data['timestamp'].min()} → {data['timestamp'].max()}")
    st.write("**Records per year**")
    yearly = ( data.groupby(data["timestamp"].dt.year) .size().rename("Records") .reset_index())
    yearly.columns = ["Year", "Records"]
    st.dataframe(yearly, use_container_width=True, hide_index=True)
    st.write("**Missing values by column**")
    missing = raw.isna().sum().sort_values(ascending=False).rename("Missing").to_frame()
    st.dataframe(missing, use_container_width=True)
    st.write("**Numeric summary**")
    st.dataframe(data[[c for c in POLLUTANTS + ["aqi"] if c in data.columns]].describe().T.round(3), use_container_width=True)
    if show_all:
        st.subheader("Filtered records")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_aqi_data.csv", mime="text/csv")

st.caption("AQI category thresholds are shown for interpretation. This dashboard is an analytical prototype; dataset provenance and measurement units should be verified before operational use.")
