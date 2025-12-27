import streamlit as st
import polars as pl
import plotly.express as px
import datetime
import os
from huggingface_hub import hf_hub_download

# ================= CONFIGURATION =================
REPO_ID = "ddechamb/sbb-2025-data"
FILENAME = "sbb_master_data.parquet"

# KNOWN CONSTRUCTION DAYS (The "Blacklist")
CONSTRUCTION_DAYS = [
    datetime.date(2025, 9, 14), datetime.date(2025, 11, 23),
    datetime.date(2025, 9, 13), datetime.date(2025, 1, 19),
    datetime.date(2025, 2, 9),  datetime.date(2025, 1, 26),
    datetime.date(2025, 2, 2),  datetime.date(2025, 1, 25),
    datetime.date(2025, 1, 18), datetime.date(2025, 2, 8),
    datetime.date(2025, 2, 1),  datetime.date(2025, 6, 28)
]

st.set_page_config(page_title="SBB 2025 Intelligence", page_icon="🚄", layout="wide")

# ================= 1. ROBUST DATA DOWNLOADER =================
@st.cache_resource
def get_local_filepath():
    """
    Downloads the file ONCE to the server's local disk.
    This fixes the '429 Rate Limit' error by stopping constant network requests.
    """
    try:
        token = st.secrets.get("HF_TOKEN", None)
        if not token:
            st.warning("⚠️ HF_TOKEN not found. Download might fail if repo is private.")
            
        with st.spinner("📦 Downloading dataset to local disk (One-time setup)..."):
            # This function automatically caches the file. It won't re-download if it exists.
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type="dataset",
                token=token
            )
        return local_path
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return None

# Execution: Get the file path
file_path = get_local_filepath()
if not file_path:
    st.stop()

# Connect Polars to the LOCAL file (Fast & No Network Limits)
lf = pl.scan_parquet(file_path)

# ================= 2. AUTO-DETECT LINES (Fixes '0 Journeys') =================
st.sidebar.title("🚄 SBB Explorer")

@st.cache_data
def get_available_lines():
    """Scans the file to find the ACTUAL line names (e.g. 'IC1' vs 'IC 1')"""
    # We scan the first 1 million rows to get a representative list of lines
    return (
        lf.select("LINIEN_TEXT")
        .unique()
        .collect()
        .get_column("LINIEN_TEXT")
        .sort()
        .to_list()
    )

with st.spinner("🔍 Scanning for available train lines..."):
    try:
        all_lines = get_available_lines()
        # Default to the first two lines found (usually IC1, IC5)
        default_lines = all_lines[:2] if len(all_lines) >= 2 else all_lines
    except Exception as e:
        st.error(f"Could not read lines: {e}")
        st.stop()

# --- FILTER 1: LINES ---
st.sidebar.subheader("Select Lines")
selected_lines = st.sidebar.multiselect(
    "Choose lines to analyze:", 
    options=all_lines, 
    default=default_lines
)

if not selected_lines:
    st.warning("👈 Please select at least one line from the sidebar.")
    st.stop()

# --- FILTER 2: NEUTRALIZATION ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Neutralization")
neutralize = st.sidebar.checkbox(
    "🛡️ Neutralize Incidents",
    value=False,
    help="Exclude known 'Total Closure' days and weekends."
)

# ================= 3. DATA PROCESSING =================
# Build Query
query = lf.filter(pl.col("LINIEN_TEXT").is_in(selected_lines))

# Load into RAM
with st.spinner(f"🚀 Processing data for {', '.join(selected_lines)}..."):
    try:
        df = query.collect()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# Post-Processing
df = df.with_columns([
    pl.col("BETRIEBSTAG").cast(pl.Date),
    ((pl.col("IS_CANCELLED")) | (pl.col("DELAY_MIN") >= 10)).alias("IS_FAILURE"),
    df["ANKUNFTSZEIT"].dt.hour().alias("HOUR"),
    df["BETRIEBSTAG"].dt.month().alias("MONTH")
])

# Apply Neutralization
if neutralize:
    df = df.filter(
        (~pl.col("BETRIEBSTAG").is_in(CONSTRUCTION_DAYS)) &
        (pl.col("BETRIEBSTAG").dt.weekday() < 6)
    )
    st.sidebar.success(f"✅ Data Neutralized")

# ================= 4. DASHBOARD =================
st.title("🇨🇭 SBB 2025: System Performance")
st.markdown(f"Analysis of **{df.height:,}** journeys on **{', '.join(selected_lines)}**")

# --- KPIS ---
col1, col2, col3, col4 = st.columns(4)

total_trains = df.height
failures = df.filter(pl.col("IS_FAILURE")).height
fail_rate = (failures / total_trains * 100) if total_trains > 0 else 0
est_hours_lost = (df["DELAY_MIN"].sum() / 60) * 600 * 0.4 

col1.metric("Total Trains", f"{total_trains:,}")
col2.metric("Failure Rate (>10m)", f"{fail_rate:.1f}%", delta_color="inverse")
col3.metric("Est. Human Hours Lost", f"{int(est_hours_lost):,}")
col4.metric("View Mode", "Structural (Neutralized)" if neutralize else "Raw Reality")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📉 Trends", "🔥 Heatmap", "🛑 Worst Trains"])

with tab1:
    st.subheader("Reliability Timeline")
    daily = (
        df.group_by("BETRIEBSTAG")
        .agg([pl.len().alias("TOTAL"), pl.col("IS_FAILURE").sum().alias("FAILURES")])
        .with_columns((pl.col("FAILURES") / pl.col("TOTAL") * 100).alias("RATE"))
        .sort("BETRIEBSTAG")
    ).to_pandas()
    
    fig = px.line(daily, x="BETRIEBSTAG", y="RATE", title="Daily Failure Rate (%)", markers=False)
    fig.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="Chaos Threshold (20%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("The 'Red Streak' (Time vs. Reliability)")
    
    heatmap_data = (
        df.group_by("HOUR")
        .agg([pl.len().alias("TOTAL"), pl.col("IS_FAILURE").sum().alias("FAILURES")])
        .with_columns((pl.col("FAILURES") / pl.col("TOTAL") * 100).alias("RATE"))
        .sort("HOUR")
    ).to_pandas()
    
    fig_heat = px.bar(heatmap_data, x="HOUR", y="RATE", color="RATE", 
                      color_continuous_scale="Reds", title="Failure Probability by Hour")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Top 10 Worst Journeys")
    worst_trains = (
        df.filter(pl.col("DELAY_MIN") > 30)
        .select(["BETRIEBSTAG", "LINIEN_TEXT", "FAHRT_BEZEICHNER", "HALTESTELLEN_NAME", "DELAY_MIN"])
        .sort("DELAY_MIN", descending=True)
        .head(10)
    ).to_pandas()
    
    st.dataframe(worst_trains, use_container_width=True)
