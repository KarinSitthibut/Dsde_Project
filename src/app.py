import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ML_data_cleaning_pipeline import preprocess_df_for_ml

st.set_page_config(page_title="Bangkok Footpath Issues Dashboard", layout="wide")

# Cream theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #F9F8F6;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e8e4db;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #EFE9E3;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.08);
        border-radius: 0 20px 20px 0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #5c4033;
    }
    .js-plotly-plot {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-radius: 15px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚧 Bangkok Footpath Issues Analysis")

# Load data once
@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed/footpath_phase1.csv")
    df = df.dropna(how="all")
    df = df.rename(columns={
        "lon": "longitude",
        "lat": "latitude"
    })
    df = df.dropna(subset=["latitude", "longitude"])
    
    if "duration_hours" in df.columns:
        df["fix_duration"] = df["duration_hours"] / 24
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["last_activity"] = pd.to_datetime(df["last_activity"], errors="coerce")
        df = df.dropna(subset=["timestamp", "last_activity"])
        df["fix_duration"] = (
            (df["last_activity"] - df["timestamp"]).dt.total_seconds() / 86400
        )
    
    df = df.dropna(subset=["fix_duration"])
    df["district"] = df["district"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")
    df["comment"] = df["comment"].fillna("")
    return df

data = load_data()

# Create tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 ML Prediction"])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    MAX_ROWS = 100000
    if len(data) > MAX_ROWS:
        display_data = data.sample(MAX_ROWS, random_state=42)
    else:
        display_data = data.copy()

    st.sidebar.header("Dashboard Filters")

    min_fix = int(display_data["fix_duration"].min())
    max_fix = int(display_data["fix_duration"].max())

    fix_range = st.sidebar.slider(
        "Time-to-fix Range (days)",
        min_value=min_fix,
        max_value=max_fix,
        value=(min_fix, max_fix)
    )

    if "timestamp" in display_data.columns:
        display_data["timestamp"] = pd.to_datetime(display_data["timestamp"], errors="coerce")
        min_date = display_data["timestamp"].min().date()
        max_date = display_data["timestamp"].max().date()
        
        date_range = st.sidebar.date_input(
            "Report Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range if not isinstance(date_range, tuple) else date_range[0]
    else:
        start_date = end_date = None

    districts = sorted(display_data["district"].dropna().unique())
    districts = [d for d in districts if d != "Unknown"]
    district_options = ["All Districts"] + districts

    selected_option = st.sidebar.selectbox(
        "District",
        district_options,
        index=0
    )

    if selected_option == "All Districts":
        selected_district = districts
    else:
        selected_district = [selected_option]

    map_style = "road"
    bandwidth = 0.006

    filtered = display_data[
        (display_data["fix_duration"] >= fix_range[0]) &
        (display_data["fix_duration"] <= fix_range[1]) &
        (display_data["district"].isin(selected_district))
    ].copy()

    if start_date is not None and end_date is not None:
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date) &
            (filtered["timestamp"].dt.date <= end_date)
        ]

    if len(filtered) == 0:
        st.warning("⚠️ No data matches the selected filters")
    else:
        RENDER_LIMIT = 20000
        map_data = filtered

        if len(map_data) > RENDER_LIMIT:
            map_data = map_data.sample(RENDER_LIMIT, random_state=42)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Cases", len(filtered))

        with col2:
            st.metric("Avg Time-to-fix (days)", f"{filtered['fix_duration'].mean():.1f}")

        with col3:
            st.metric("Issue Types", filtered['type'].nunique())

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("⏱ Time-to-fix Distribution")
            fig_fix = px.histogram(
                filtered,
                x="fix_duration",
                nbins=50,
                title="Distribution of Time-to-fix (days)",
                color_discrete_sequence=["#0096FF"]
            )
            fig_fix.update_layout(
                showlegend=False,
                plot_bgcolor='#F9F8F6',
                paper_bgcolor='#F9F8F6'
            )
            st.plotly_chart(fig_fix, use_container_width=True)

        with chart_col2:
            st.subheader("📊 Top 10 Districts by Cases")
            top_districts = filtered['district'].value_counts().head(10)
            fig_bar = px.bar(
                x=top_districts.values,
                y=top_districts.index,
                orientation='h',
                title="Top 10 Districts with Most Issues",
                labels={'x': 'Number of Cases', 'y': 'District'},
                color=top_districts.values,
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(
                showlegend=False, 
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor='#F9F8F6',
                paper_bgcolor='#F9F8F6'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.header("🗺 Footpath Issues Map")

        point_layer = pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position=["longitude", "latitude"],
            get_color=[0, 150, 255, 160],
            get_radius=15,
            pickable=True
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[point_layer],
                initial_view_state=pdk.ViewState(
                    latitude=map_data["latitude"].mean(),
                    longitude=map_data["longitude"].mean(),
                    zoom=13
                ),
                map_style=map_style
            ),
            height=600
        )

        st.header("🌡 Density Estimation")

        try:
            kde_sample_size = min(10000, len(filtered))
            kde_data = filtered.sample(kde_sample_size, random_state=42)
            
            coords = kde_data[["latitude", "longitude"]].values
            kde = KernelDensity(bandwidth=bandwidth).fit(coords)
            
            density = np.exp(kde.score_samples(coords))
            kde_data = kde_data.copy()
            kde_data["density"] = density
            
            density_range = density.max() - density.min()
            if density_range == 0 or np.isnan(density_range):
                kde_data["density_norm"] = 0.5
            else:
                kde_data["density_norm"] = (density - density.min()) / density_range
            
            kde_viz = kde_data.sample(min(RENDER_LIMIT, len(kde_data)), random_state=42).copy()
            
            kde_viz["color"] = kde_viz["density_norm"].apply(
                lambda x: [int(255*x), int(255*(1-x)), 80, 160]
            )
            
            kde_layer = pdk.Layer(
                "ScatterplotLayer",
                kde_viz,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius=15,
                pickable=True
            )
            
            st.pydeck_chart(
                pdk.Deck(
                    layers=[kde_layer],
                    initial_view_state=pdk.ViewState(
                        latitude=kde_data["latitude"].mean(),
                        longitude=kde_data["longitude"].mean(),
                        zoom=13
                    ),
                    map_style=map_style
                ),
                height=600
            )

        except Exception as e:
            st.error(f"KDE Error: {e}")

        st.header("📋 Detailed Data Table")

        show_rows = st.selectbox("Show rows", [10, 25, 50, 100], index=1)

        table_data = filtered.copy()
        if "timestamp" in table_data.columns:
            table_data = table_data.sort_values(by="timestamp", ascending=False)

        table_data = table_data.head(show_rows)

        display_columns = {
            "ticket_id": "Ticket ID",
            "type": "Issue Type",
            "comment": "Comment",
            "address": "Address",
            "fix_duration": "Days to Fix",
            "timestamp": "Reported",
            "last_activity": "Completed"
        }

        available_columns = {k: v for k, v in display_columns.items() if k in table_data.columns}

        table_display = table_data[list(available_columns.keys())].copy()
        table_display = table_display.rename(columns=available_columns)

        if "Days to Fix" in table_display.columns:
            table_display["Days to Fix"] = table_display["Days to Fix"].round(1)

        if "Reported" in table_display.columns:
            table_display["Reported"] = pd.to_datetime(table_display["Reported"]).dt.strftime('%Y-%m-%d')

        if "Completed" in table_display.columns:
            table_display["Completed"] = pd.to_datetime(table_display["Completed"]).dt.strftime('%Y-%m-%d')

        if "Comment" in table_display.columns:
            table_display["Comment"] = table_display["Comment"].str[:100] + "..."

        st.dataframe(
            table_display,
            use_container_width=True,
            height=400,
            hide_index=True
        )

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name=f"footpath_issues_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ==================== TAB 2: ML PREDICTION ====================
with tab2:
    st.header("🤖 Machine Learning Duration Prediction")
    
    # Prepare data for ML
    ml_data = data.copy()
    ml_data = ml_data.rename(columns={"longitude": "lon", "latitude": "lat"})
    
    st.write(f"Total rows in dataset: {len(ml_data)}")
    
    N_EST = 100

    # Cached training function
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

    # Train model
    start = time.perf_counter()
    with st.spinner("Training model (cached)..."):
        pipe, model, X_train, X_test, y_train, y_test = train_model_cached(ml_data)
    elapsed = time.perf_counter() - start
    st.success(f"✅ Training completed in {elapsed:.1f} seconds (cached)")

    # Evaluation
    y_pred_log = model.predict(X_test)
    y_test_hours = np.expm1(y_test)
    y_pred_hours = np.expm1(y_pred_log)

    col1, col2, col3 = st.columns(3)
    
    mae = mean_absolute_error(y_test, y_pred_log)
    mse = mean_squared_error(y_test, y_pred_log)
    rmse = np.sqrt(mse)
    
    with col1:
        st.metric("MAE (log target)", f"{mae:.4f}")
    with col2:
        st.metric("MSE (log target)", f"{mse:.4f}")
    with col3:
        st.metric("RMSE (log target)", f"{rmse:.4f}")

    # Plot
    st.subheader("📈 Actual vs Predicted Duration")
    order = np.argsort(y_test_hours.values)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test_hours.values[order], label="Actual", linewidth=2)
    ax.plot(y_pred_hours[order], label="Predicted", alpha=0.8)
    ax.set_xlabel("Sample index (sorted by actual duration)")
    ax.set_ylabel("Duration (hours)")
    ax.set_title("Actual vs Predicted Duration (sorted)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Single prediction form
    st.subheader("🔮 Single Report Prediction")
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        ts = st.text_input("Timestamp", "2025-12-05 10:30:00")
        comment = st.text_area("Comment", "ทางเท้าชำรุด หน้าบ้านเลขที่ 123")
        lon = st.number_input("Longitude", value=100.5018, format="%.6f")
        lat = st.number_input("Latitude", value=13.7563, format="%.6f")
    
    with pred_col2:
        sub = st.text_input("Subdistrict", "บางกอกใหญ่")
        typ = st.text_input("Type", "{ถนน,ทางเท้า}")
        
        if st.button("🎯 Predict Duration", type="primary"):
            single_raw = pd.DataFrame([{
                "timestamp": ts, 
                "comment": comment, 
                "lon": lon, 
                "lat": lat,
                "subdistrict": sub, 
                "type": typ
            }])
            
            try:
                # Preprocess single row
                single_X, _ = preprocess_df_for_ml(single_raw)

                # Align to training columns
                ref = X_train.columns
                aligned = pd.DataFrame(0, index=single_X.index, columns=ref)
                common = single_X.columns.intersection(ref)
                if len(common) > 0:
                    aligned.loc[:, common] = single_X.loc[:, common].values

                # Fill datetime features
                try:
                    ts_val = pd.to_datetime(ts)
                    dt = {
                        'year': ts_val.year, 
                        'month': ts_val.month, 
                        'day': ts_val.day,
                        'weekday': ts_val.weekday(), 
                        'hour': ts_val.hour, 
                        'is_weekend': int(ts_val.weekday() >= 5)
                    }
                    for k, v in dt.items():
                        if k in aligned.columns:
                            aligned.loc[:, k] = v
                except Exception:
                    pass

                # Fill coordinates
                if 'lon' in aligned.columns: 
                    aligned.loc[:, 'lon'] = float(lon)
                if 'lat' in aligned.columns: 
                    aligned.loc[:, 'lat'] = float(lat)

                aligned = aligned.astype(float)
                ylog = model.predict(aligned)
                yhours = float(np.expm1(ylog)[0])
                ydays = yhours / 24
                
                st.success(f"### Predicted Duration")
                st.metric("Hours", f"{yhours:.1f}")
                st.metric("Days", f"{ydays:.1f}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")