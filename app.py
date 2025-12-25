import streamlit as st
import polars as pl
import plotly.express as px
import datetime

# ================= CONFIGURATION =================
# If your parquet file is < 100MB, keep it in the repo.
# If larger, you might need to host it elsewhere (S3) or use Git LFS.
DATA_PATH = 'sbb_master_data.parquet'

# KNOWN CONSTRUCTION DAYS (The "Blacklist")
# Dates where the system was structurally broken (Planned Works)
CONSTRUCTION_DAYS = [
    datetime.date(2025, 9, 14), datetime.date(2025, 11, 23),
    datetime.date(2025, 9, 13), datetime.date(2025, 1, 19),
    datetime.date(2025, 2, 9),  datetime.date(2025, 1, 26),
    datetime.date(2025, 2, 2),  datetime.date(2025, 1, 25),
    datetime.date(2025, 1, 18), datetime.date(2025, 2, 8),
    datetime.date(2025, 2, 1),  datetime.date(2025, 6, 28) # Start of Summer works
]

st.set_page_config(page_title="SBB 2025 Intelligence", page_icon="🚄", layout="wide")

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    try:
        df = pl.read_parquet(DATA_PATH)
        # Ensure dates are Date objects, not strings
        df = df.with_columns([
            pl.col("BETRIEBSTAG").cast(pl.Date),
            ((pl.col("IS_CANCELLED")) | (pl.col("DELAY_MIN") >= 10)).alias("IS_FAILURE"),
            df["ANKUNFTSZEIT"].dt.hour().alias("HOUR"),
            df["BETRIEBSTAG"].dt.month().alias("MONTH")
        ])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("🚄 SBB Explorer")
st.sidebar.markdown(f"**Data:** {df.height:,} trains (IC1/IC5)")

# --- FILTER 1: LINES ---
lines = df["LINIEN_TEXT"].unique().sort().to_list()
selected_lines = st.sidebar.multiselect("Select Lines", lines, default=lines)

# --- FILTER 2: NEUTRALIZATION (The Belgian Feature) ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Neutralization")
neutralize = st.sidebar.checkbox(
    "🛡️ Neutralize Incidents",
    value=False,
    help="Exclude known 'Total Closure' days and weekends to show underlying structural performance."
)

# --- FILTERING LOGIC ---
filtered_df = df.filter(pl.col("LINIEN_TEXT").is_in(selected_lines))

if neutralize:
    # Remove the Blacklisted Dates AND Weekends (Sat=6, Sun=7)
    filtered_df = filtered_df.filter(
        (~pl.col("BETRIEBSTAG").is_in(CONSTRUCTION_DAYS)) &
        (pl.col("BETRIEBSTAG").dt.weekday() < 6)
    )
    st.sidebar.success(f"✅ Major incidents & weekends excluded.")

# ================= MAIN DASHBOARD =================
st.title("🇨🇭 SBB 2025: System Performance")
st.markdown(f"Analysis of **{filtered_df.height:,}** intercity journeys.")

# --- KPIS ---
col1, col2, col3, col4 = st.columns(4)

total_trains = filtered_df.height
failures = filtered_df.filter(pl.col("IS_FAILURE")).height
fail_rate = (failures / total_trains * 100) if total_trains > 0 else 0

# Estimated Hours Lost (Simplified Calc for Speed)
# Avg Train Capacity * Load Factor (0.4) * Delay Hours
est_hours_lost = (filtered_df["DELAY_MIN"].sum() / 60) * 600 * 0.4 

col1.metric("Total Trains", f"{total_trains:,}")
col2.metric("Failure Rate (>10m)", f"{fail_rate:.1f}%", delta_color="inverse")
col3.metric("Est. Human Hours Lost", f"{int(est_hours_lost):,}")
col4.metric("Dataset Status", "Neutralized" if neutralize else "Raw Reality")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📉 Trends", "🔥 Heatmap", "🛑 Worst Trains"])

with tab1:
    st.subheader("Reliability Timeline")
    # Daily aggregation
    daily = (
        filtered_df.group_by("BETRIEBSTAG")
        .agg([pl.len().alias("TOTAL"), pl.col("IS_FAILURE").sum().alias("FAILURES")])
        .with_columns((pl.col("FAILURES") / pl.col("TOTAL") * 100).alias("RATE"))
        .sort("BETRIEBSTAG")
    ).to_pandas()
    
    fig = px.line(daily, x="BETRIEBSTAG", y="RATE", title="Daily Failure Rate (%)", markers=False)
    # Add a red line for the "Chaos Threshold"
    fig.add_hline(y=20, line_dash="dot", line_color="red", annotation_text="Chaos Threshold (20%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("The 'Red Streak' (Time vs. Reliability)")
    st.markdown("Darker Red = Higher probability of failure at this hour.")
    
    # Hourly Heatmap Data
    heatmap_data = (
        filtered_df.group_by("HOUR")
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
        filtered_df.filter(pl.col("DELAY_MIN") > 30)
        .select(["BETRIEBSTAG", "LINIEN_TEXT", "FAHRT_BEZEICHNER", "HALTESTELLEN_NAME", "DELAY_MIN"])
        .sort("DELAY_MIN", descending=True)
        .head(10)
    ).to_pandas()
    
    st.dataframe(worst_trains, use_container_width=True)
