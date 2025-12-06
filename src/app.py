import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime
from sklearn.neighbors import KernelDensity

st.set_page_config(page_title="Bangkok Footpath Issues Dashboard", layout="wide")
st.title("🚧 Bangkok Footpath Issues Analysis")

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

# Load Dataset
data = load_data()

MAX_ROWS = 100000
if len(data) > MAX_ROWS:
    data = data.sample(MAX_ROWS, random_state=42)

# Sidebar Filters
st.sidebar.header("Filters")

min_fix = int(data["fix_duration"].min())
max_fix = int(data["fix_duration"].max())

fix_range = st.sidebar.slider(
    "Time-to-fix Range (days)",
    min_value=min_fix,
    max_value=max_fix,
    value=(min_fix, max_fix)
)

districts = sorted(data["district"].dropna().unique())


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

# Apply sidebar filters
filtered = data[
    (data["fix_duration"] >= fix_range[0]) &
    (data["fix_duration"] <= fix_range[1]) &
    (data["district"].isin(selected_district))
].copy()

# Check if filtered data is empty
if len(filtered) == 0:
    st.warning("⚠️ No data matches the selected filters")
    st.stop()


RENDER_LIMIT = 20000
map_data = filtered

if len(map_data) > RENDER_LIMIT:
    map_data = map_data.sample(RENDER_LIMIT, random_state=42)

# Metrics
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
    fig_fix.update_layout(showlegend=False)
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
    fig_bar.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# All Issue Points Map
st.header("📍 Footpath Issues Map")

point_layer = pdk.Layer(
    "ScatterplotLayer",
    map_data,
    get_position=["longitude", "latitude"],
    get_color=[0, 150, 255, 160],
    get_radius=12,
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
    )
)

# KDE Density Map
st.header("🌡 Density Estimation (KDE)")

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
        get_radius=20,
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
        )
    )

except Exception as e:
    st.error(f"KDE Error: {e}")  