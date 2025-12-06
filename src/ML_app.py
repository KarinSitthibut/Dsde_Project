# streamlit_app_simple.py — retrain once per session (cached), simple UI
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ML_data_cleaning_pipeline import preprocess_df_for_ml

st.set_page_config(page_title="Footpath (simple)", layout="wide")
st.title("Footpath Duration — simple demo (cached training)")

RAW = "../data/processed/footpath_phase1.csv"
df = pd.read_csv(RAW)
st.write(f"Rows: {len(df)}")

# fixed
N_EST = 100

# cached training function — runs once per session (or until code changes)
@st.cache_resource
def train_model_cached(df, n_estimators=N_EST):
    X_full, pipe = preprocess_df_for_ml(df)
    y_log = np.log1p(df["duration_hours"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_log, test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return pipe, model, X_train, X_test, y_train, y_test

# 1) train (cached) and measure elapsed time
start = time.perf_counter()
with st.spinner("Fitting pipeline and training model (cached)..."):
    pipe, model, X_train, X_test, y_train, y_test = train_model_cached(df)
elapsed = time.perf_counter() - start
st.success(f"Training completed in {elapsed:.1f} seconds (cached)")

# 2) evaluation & sorted plot
y_pred_log = model.predict(X_test)
y_test_hours = np.expm1(y_test)
y_pred_hours = np.expm1(y_pred_log)

order = np.argsort(y_test_hours.values)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test_hours.values[order], label="Actual", linewidth=2)
ax.plot(y_pred_hours[order], label="Predicted", alpha=0.8)
ax.set_xlabel("Sample index (sorted by actual duration)")
ax.set_ylabel("Duration (hours)")
ax.set_title("Actual vs Predicted Duration (sorted)")
ax.legend(); ax.grid(True)
st.pyplot(fig)

# show simple metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred_log)
mse = mean_squared_error(y_test, y_pred_log)
rmse = np.sqrt(mse)
st.metric("MAE (log target)", f"{mae:.4f}")
st.metric("MSE (log target)", f"{mse:.4f}")
st.metric("RMSE (log target)", f"{rmse:.4f}")

# 3) single manual prediction (simple form)
st.sidebar.header("Manual single report")
ts = st.sidebar.text_input("timestamp", "2025-12-05 10:30:00")
comment = st.sidebar.text_area("comment", "ทางเท้าชำรุด หน้าบ้านเลขที่ 123")
lon = st.sidebar.number_input("lon", value=100.5018, format="%.6f")
lat = st.sidebar.number_input("lat", value=13.7563, format="%.6f")
sub = st.sidebar.text_input("subdistrict", "บางกอกใหญ่")
typ = st.sidebar.text_input("type", "{ถนน,ทางเท้า}")
if st.sidebar.button("Predict single"):
    single_raw = pd.DataFrame([{
        "timestamp": ts, "comment": comment, "lon": lon, "lat": lat,
        "subdistrict": sub, "type": typ
    }])
    # preprocess single row (fits pipeline on the single row as you requested)
    single_X, _ = preprocess_df_for_ml(single_raw)

    # align to training columns (fill missing with 0)
    ref = X_train.columns
    aligned = pd.DataFrame(0, index=single_X.index, columns=ref)
    common = single_X.columns.intersection(ref)
    if len(common) > 0:
        aligned.loc[:, common] = single_X.loc[:, common].values

    # ensure fixed datetime cols exist, fill from timestamp if missing
    try:
        ts_val = pd.to_datetime(ts)
        dt = {'year': ts_val.year, 'month': ts_val.month, 'day': ts_val.day,
              'weekday': ts_val.weekday(), 'hour': ts_val.hour, 'is_weekend': int(ts_val.weekday() >= 5)}
        for k, v in dt.items():
            if k in aligned.columns:
                aligned.loc[:, k] = v
    except Exception:
        pass

    # lon/lat
    if 'lon' in aligned.columns: aligned.loc[:, 'lon'] = float(lon)
    if 'lat' in aligned.columns: aligned.loc[:, 'lat'] = float(lat)

    aligned = aligned.astype(float)
    ylog = model.predict(aligned)
    yhours = float(np.expm1(ylog)[0])
    st.sidebar.success(f"Predicted duration_hours = {yhours:.3f}")
