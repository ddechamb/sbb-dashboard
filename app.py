import streamlit as st
import polars as pl
import plotly.express as px
import datetime
import pyarrow.dataset as ds
import fsspec

# ================= CONFIGURATION =================
# URL to your Parquet file on Hugging Face (Resolve URL)
DATA_URL = "https://huggingface.co/datasets/ddechamb/sbb-2025-data/resolve/main/sbb_master_data.parquet"

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

# ================= LAZY DATA LOADER (PyArrow Method) =================
@st.cache_resource
def get_lazy_frame():
    """
    Connects to Hugging Face using PyArrow + FSSPEC.
    This is the most robust way to handle authentication without crashing Polars.
    """
    try:
        # 1. Prepare Authentication Headers
        if "HF_TOKEN" in st.secrets:
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        else:
            st.warning("⚠️ No HF_TOKEN found. Using anonymous access (might fail).")
            headers = {}

        # 2. Create a virtual filesystem connection
        fs = fsspec.filesystem("http", headers=headers)

        # 3. Create a PyArrow Dataset (Lazy)
        dataset = ds.dataset(DATA_URL, filesystem=fs, format="parquet")

        # 4. Convert to Polars LazyFrame
        return pl.scan_pyarrow_dataset(dataset)

    except Exception as e:
        st.error(f"Failed to connect to data: {e}")
        return None

# Initialize connection
lf = get_lazy_frame()

if lf is None:
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("🚄 SBB Explorer")

# --- FILTER 1: LINES ---
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
    help="Exclude known 'Total Closure' days and weekends."
)

# ================= DATA PROCESSING =================
# 1. Build the Query
query = lf.filter(pl.col("LINIEN_TEXT").is_in(selected_lines))

# 2. Collect into RAM
with st.spinner(f"Downloading data for {', '.join(selected_lines)}..."):
    try:
        df = query.collect()
    except Exception as e:
        st.error(f"Error streaming data: {e}")
        st.error("Check your HF_TOKEN in Streamlit Secrets.")
        st.stop()

# 3. Post-Processing
df = df.with_columns([
    pl.col("BETRIEBSTAG").cast(pl.Date),
    ((pl.col("IS_CANCELLED")) | (pl.col("DELAY_MIN") >= 10)).alias("IS_FAILURE"),
    df["ANKUNFTSZEIT"].dt.hour().alias("HOUR"),
    df["BETRIEBSTAG"].dt.month().alias("MONTH")
])

# 4. Apply Neutralization
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
    st.markdown("Darker Red = Higher probability of failure at this hour.")
    
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
