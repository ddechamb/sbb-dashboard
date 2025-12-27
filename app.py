import streamlit as st
import polars as pl
import plotly.express as px
import datetime
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

# ================= CACHED DATA DOWNLOADER =================
@st.cache_resource
def download_data():
    """
    Downloads the 5GB file to the local Streamlit disk cache.
    This avoids the '429 Too Many Requests' error by making 1 request instead of 5000.
    """
    try:
        # Get token from secrets
        token = st.secrets.get("HF_TOKEN", None)
        if not token:
            st.warning("⚠️ No HF_TOKEN found. Download might fail if repo is private/gated.")

        with st.spinner("Downloading dataset to local cache (this happens once)..."):
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type="dataset",
                token=token
            )
        return local_path
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# 1. Download/Locate the file
file_path = download_data()
if not file_path:
    st.stop()

# 2. Connect LazyFrame to local file (Zero RAM usage so far)
lf = pl.scan_parquet(file_path)

# ================= SIDEBAR =================
st.sidebar.title("🚄 SBB Explorer")

# --- FILTER 1: LINES ---
common_lines = ["IC 1", "IC 5", "IC 3", "IC 6", "IC 8", "IC 61", "IR 15", "IR 90", "EC", "IC 2"]
selected_lines = st.sidebar.multiselect("Select Lines", common_lines, default=["IC 1", "IC 5"])

if not selected_lines:
    st.warning("Please select at least one line.")
    st.stop()

# --- FILTER 2: NEUTRALIZATION ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Neutralization")
neutralize = st.sidebar.checkbox(
    "🛡️ Neutralize Incidents",
    value=False,
    help="Exclude known 'Total Closure' days and weekends."
)

# ================= DATA PROCESSING =================
# 3. Build Query
query = lf.filter(pl.col("LINIEN_TEXT").is_in(selected_lines))

# 4. Load into RAM
with st.spinner(f"Processing data for {', '.join(selected_lines)}..."):
    try:
        df = query.collect()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# 5. Post-Processing
df = df.with_columns([
    pl.col("BETRIEBSTAG").cast(pl.Date),
    ((pl.col("IS_CANCELLED")) | (pl.col("DELAY_MIN") >= 10)).alias("IS_FAILURE"),
    df["ANKUNFTSZEIT"].dt.hour().alias("HOUR"),
    df["BETRIEBSTAG"].dt.month().alias("MONTH")
])

# 6. Apply Neutralization
if neutralize:
    df = df.filter(
        (~pl.col("BETRIEBSTAG").is_in(CONSTRUCTION_DAYS)) &
        (pl.col("BETRIEBSTAG").dt.weekday() < 6)
    )
    st.sidebar.success(f"✅ Data Neutralized")

# ================= MAIN DASHBOARD =================
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
