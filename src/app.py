import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime
from sklearn.cluster import DBSCAN
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

# --------------------------
# Sidebar Filters
# --------------------------
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
selected_district = st.sidebar.multiselect(
    "Districts",
    districts,
    default=districts
)

# DBSCAN Parameters
st.sidebar.header("DBSCAN Parameters")
eps_degrees = st.sidebar.slider("eps (degrees)", 0.0005, 0.005, 0.0015, 0.0005)
min_samples = st.sidebar.slider("min_samples", 2, 15, 5)
num_top_clusters = st.sidebar.slider("Top Clusters", 1, 10, 5)

# KDE
st.sidebar.header("KDE Parameters")
bandwidth = st.sidebar.slider("KDE Bandwidth", 0.001, 0.020, 0.006, 0.001)

# MAP
map_style = st.sidebar.selectbox("Map Style", ["Dark", "Light", "Road", "Satellite"])
MAP_STYLES = {"Dark": "dark", "Light": "light", "Road": "road", "Satellite": "satellite"}

# --------------------------
# Apply sidebar filters
# --------------------------
filtered = data[
    (data["fix_duration"] >= fix_range[0]) &
    (data["fix_duration"] <= fix_range[1]) &
    (data["district"].isin(selected_district))
].copy()

# --------------------------
# Map Rendering Limit
# IMPORTANT → prevents MessageSizeError
# --------------------------

RENDER_LIMIT = 20000
map_data = filtered

if len(map_data) > RENDER_LIMIT:
    map_data = map_data.sample(RENDER_LIMIT, random_state=42)
    ##st.info(f"Map limited to {RENDER_LIMIT:,} points to avoid browser memory errors.")

# --------------------------
# Metrics
# --------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Cases", len(filtered))

with col2:
    st.metric("Avg Time-to-fix (days)", f"{filtered['fix_duration'].mean():.1f}")

with col3:
    st.metric("Issue Types", filtered['type'].nunique())

st.header("⏱ Time-to-fix Distribution")

fig_fix = px.histogram(
    filtered,
    x="fix_duration",
    nbins=50,
    title="Distribution of Time-to-fix (days)"
)
st.plotly_chart(fig_fix, use_container_width=True)


st.header("🔥 Hotspot Clustering")

try:
    coords = filtered[["latitude", "longitude"]]

    dbscan = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)
    filtered["cluster"] = dbscan.labels_

    cluster_counts = filtered["cluster"].value_counts()
    cluster_counts = cluster_counts[cluster_counts.index != -1]

    top_clusters = cluster_counts.head(num_top_clusters).index

    viz_data = filtered[filtered["cluster"].isin(top_clusters)]

    # Limit for safe map rendering
    viz_data = viz_data.sample(min(RENDER_LIMIT, len(viz_data)), random_state=42)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        viz_data,
        get_position=["longitude", "latitude"],
        get_color=[255, 100, 100, 160],
        get_radius=10,
        pickable=True
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[scatter],
            initial_view_state=pdk.ViewState(
                latitude=map_data["latitude"].mean(),
                longitude=map_data["longitude"].mean(),
                zoom=11
            ),
            map_style=MAP_STYLES[map_style]
        )
    )

except Exception as e:
    st.error(f"DBSCAN Error: {e}")


st.header("🌡 Density Estimation")

try:
    coords = filtered[["latitude", "longitude"]].values
    kde = KernelDensity(bandwidth=bandwidth).fit(coords)

    density = np.exp(kde.score_samples(coords))
    filtered["density"] = density
    filtered["density_norm"] = (density - density.min()) / (density.max() - density.min())

    filtered["color"] = filtered["density_norm"].apply(
        lambda x: [int(255*x), int(255*(1-x)), 80, 160]
    )

    kde_viz = filtered.sample(min(RENDER_LIMIT, len(filtered)), random_state=42)

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
                latitude=map_data["latitude"].mean(),
                longitude=map_data["longitude"].mean(),
                zoom=11
            ),
            map_style=MAP_STYLES[map_style]
        )
    )

except Exception as e:
    st.error(f"KDE Error: {e}")
