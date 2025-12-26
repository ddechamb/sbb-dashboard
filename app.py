import streamlit as st
import polars as pl
import plotly.express as px
import datetime

# ================= CONFIGURATION =================
# URL to your Parquet file on Hugging Face
DATA_URL = "https://huggingface.co/datasets/ddechamb/sbb-2025-data/resolve/main/sbb_master_data.parquet"

# KNOWN CONSTRUCTION DAYS (The "Blacklist")
# Dates where the system was structurally broken (Planned Works)
CONSTRUCTION_DAYS = [
    datetime.date(2025, 9, 14), datetime.date(2025, 11, 23),
    datetime.date(2025, 9, 13), datetime.date(2025, 1, 19),
    datetime.date(2025, 2, 9),  datetime.date(2025, 1, 26),
    datetime.date(2025, 2, 2),  datetime.date(2025, 1, 25),
    datetime.date(2025, 1, 18), datetime.date(2025, 2, 8),
    datetime.date(2025, 2, 1),  datetime.date(2025, 6, 28)
]

st.set_page_config(page_title="SBB 2025 Intelligence", page_icon="🚄", layout="wide")

# ================= LAZY DATA LOADER =================
@st.cache_resource
def get_lazy_frame():
    """
    Connects to the parquet file without downloading it.
    Uses st.secrets['HF_TOKEN'] to authenticate and avoid Rate Limits.
    """
    try:
        # Check if token exists in secrets
        if "HF_TOKEN" in st.secrets:
            storage_options = {"token": st.secrets["HF_TOKEN"]}
        else:
            st.warning("⚠️ No HF_TOKEN found in secrets. You might hit rate limits.")
            storage_options = None
            
        return pl.scan_parquet(DATA_URL, storage_options=storage_options)
    except Exception as e:
        st.error(f"Failed to connect to data: {e}")
        return None

lf = get_lazy_frame()

if lf is None:
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("🚄 SBB Explorer")

# --- FILTER 1: LINES ---
# Pre-defined list to avoid scanning 5GB file for unique values
common_lines = ["IC 1", "IC 5", "IC 3", "IC 6", "IC 8", "IC 61", "IR 15", "IR 90", "EC", "IC 2"]
selected_lines = st.sidebar.multiselect("Select Lines", common_lines, default=["IC 1", "IC 5"])

if not selected_lines:
    st.warning("Please select at least one line to load data.")
    st.stop()

# --- FILTER 2: NEUTRALIZATION ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Neutralization")
neutralize = st.sidebar.checkbox(
    "🛡️ Neutralize Incidents",
    value=False,
    help="Exclude known 'Total Closure' days and weekends to show underlying structural performance."
)

# ================= DATA PROCESSING =================
# 1. Build the Query (Lazy)
query = lf.filter(pl.col("LINIEN_TEXT").is_in(selected_lines))

# 2. Collect into RAM
# This is the "Expensive" step where data actually downloads
with st.spinner(f"Downloading data for {', '.join(selected_lines)}..."):
    try:
        df = query.collect()
    except Exception as e:
        st.error(f"Error streaming data: {e}")
        st.error("Did you add your HF_TOKEN to Streamlit Secrets?")
        st.stop()

# 3. Post-Processing (In Memory)
df = df.with_columns([
    pl.col("BETRIEBSTAG").cast(pl.Date),
    ((pl.col("IS_CANCELLED")) | (pl.col("DELAY_MIN") >= 10)).alias("IS_FAILURE"),
    df["ANKUNFTSZEIT"].dt.hour().alias("HOUR"),
    df["BETRIEBSTAG"].dt.month().alias("MONTH")
])

# 4. Apply Neutralization Filter
if neutralize:
    df = df.filter(
        (~pl.col("BETRIEBSTAG").is_in(CONSTRUCTION_DAYS)) &
        (pl.col("BETRIEBSTAG").dt.weekday() < 6)
    )
    st.sidebar.success(f"✅ Data Neutralized (Weekends & Incidents removed)")

# ================= MAIN DASHBOARD =================
st.title("🇨🇭 SBB 2025: System Performance")
st.markdown(f"Analysis of **{df.height:,}** journeys on **{', '.join(selected_lines)}**")

# --- KPIS ---
col1, col2, col3, col4 = st.columns(4)

total_trains = df.height
failures = df.filter(pl.col("IS_FAILURE")).height
fail_rate = (failures / total_trains * 100) if total_trains > 0 else 0
# Estimated Hours Lost: (Total Delay Mins / 60) * Avg Capacity (600) * Load Factor (0.4)
est_hours_lost = (df["DELAY_MIN"].sum() / 60) * 600 * 0.4 

col1.metric("Total Trains", f"{total_trains:,}")
col2.metric("Failure Rate (>10m)", f"{fail_rate:.1f}%", delta_color="inverse")
col3.metric("Est. Human Hours Lost", f"{int(est_hours_lost):,}")
col4.metric("View Mode", "Structural (Neutralized)" if neutralize else "Raw Reality")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📉 Trends", "🔥 Heatmap", "🛑 Worst Trains"])

with tab1:
    st.subheader("Reliability Timeline")
    # Daily aggregation
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
    st.markdown("Darker Red = Higher probability of failure at this hour.")
    
    # Hourly Heatmap Data
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
    st.write("The specific trains that caused the most pain in this filtered view.")
    
    worst_trains = (
        df.filter(pl.col("DELAY_MIN") > 30)
        .select(["BETRIEBSTAG", "LINIEN_TEXT", "FAHRT_BEZEICHNER", "HALTESTELLEN_NAME", "DELAY_MIN"])
        .sort("DELAY_MIN", descending=True)
        .head(10)
    ).to_pandas()
    
    st.dataframe(worst_trains, use_container_width=True)
