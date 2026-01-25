import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# Import Core Logic
# Note: We import directly from modules to ensure safety against __init__ typos
from risklab_core.market_data.sources import YahooFinanceSource
from risklab_core.contracts.market_data import (
    PriceRequest, 
    ReturnsSpec, 
    ReSampleSpec, 
    OutlierSpec,
    AlignSpec
)
from risklab_core.market_data.transforms import (
    to_returns, 
    resample_prices, 
    align_assets
)
from risklab_core.market_data.outliers import handle_outliers

# Page Config
st.set_page_config(
    page_title="RiskLab Explorer",
    page_icon="lab",
    layout="wide"
)

st.title("RiskLab — Market Data Pipeline")
st.markdown("""
This interface demonstrates the **RiskLab Core** pipeline:  
`Source` -> `Resample` -> `Outliers` -> `Returns`
""")

# --- Sidebar: Data Source Configuration ---
with st.sidebar:
    st.header("1. Data Source")
    
    # Symbols Input
    default_symbols = "SPY, QQQ, GLD, TLT"
    symbols_str = st.text_input("Tickers (comma separated)", value=default_symbols)
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    
    # Date Range
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = col2.date_input("End Date", value=date.today())
    
    # Field Selection
    field = st.selectbox(
        "Field", 
        ["adj_close", "close", "open", "high", "low", "volume"], 
        index=0
    )
    
    load_btn = st.button("Load Market Data", type="primary")

# --- Main Logic ---

# Initialize Session State for Data
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None

if load_btn and symbols:
    with st.spinner(f"Fetching data for {len(symbols)} symbols..."):
        try:
            # 1. Instantiate Source
            src = YahooFinanceSource()
            
            # 2. Create Request
            req = PriceRequest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                field=field
            )
            
            # 3. Fetch Data
            df = src.get_prices(req)
            
            if df.empty:
                st.error("No data returned. Check your symbols or date range.")
            else:
                st.session_state.raw_data = df
                st.success(f"Loaded {len(df)} rows for {len(symbols)} assets.")
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Proceed if we have data
if st.session_state.raw_data is not None:
    df_raw = st.session_state.raw_data
    
    # --- Tab Layout ---
    tab_raw, tab_pipeline = st.tabs(["Raw Data", "Pipeline Config"])
    
    with tab_raw:
        st.subheader("Raw Prices")
        st.dataframe(df_raw.tail())
        
        # Plot Raw
        fig = px.line(df_raw, title=f"Raw Data ({field})")
        st.plotly_chart(fig, use_container_width=True)

    with tab_pipeline:
        st.sidebar.header("2. Pipeline Transforms")
        
        # --- Resampling Spec ---
        st.sidebar.subheader("Resampling")
        do_resample = st.sidebar.checkbox("Enable Resampling", value=False)
        rs_rule = st.sidebar.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=1)
        rs_how = st.sidebar.selectbox("Method", ["last", "mean", "first"], index=0)
        
        # --- Outliers Spec ---
        st.sidebar.subheader("Outlier Handling")
        outlier_method = st.sidebar.selectbox("Method", ["none", "winsorize", "clip"], index=0)
        
        outlier_params = {}
        if outlier_method == "winsorize":
            lq = st.sidebar.slider("Lower Quantile", 0.0, 0.1, 0.01)
            uq = st.sidebar.slider("Upper Quantile", 0.9, 1.0, 0.99)
            outlier_params = {"lower_q": lq, "upper_q": uq}
        elif outlier_method == "clip":
            cl = st.sidebar.number_input("Clip Low", value=None)
            ch = st.sidebar.number_input("Clip High", value=None)
            outlier_params = {"clip_low": cl, "clip_high": ch}

        # --- Returns Spec ---
        st.sidebar.subheader("Returns Calculation")
        calc_returns = st.sidebar.checkbox("Calculate Returns", value=True)
        ret_method = st.sidebar.radio("Returns Type", ["simple", "log"], index=1)

        # --- Apply Pipeline ---
        df_processed = df_raw.copy()
        
        # 1. Resample
        if do_resample:
            spec = ReSampleSpec(rule=rs_rule, how=rs_how)
            df_processed = resample_prices(df_processed, spec)
            
        # 2. Outliers
        if outlier_method != "none":
            # Map None/String "none" correctly for Pydantic
            method_val = None if outlier_method == "none" else outlier_method
            spec = OutlierSpec(method=method_val, **outlier_params)
            df_processed = handle_outliers(df_processed, spec)
            
        # 3. Returns
        if calc_returns:
            spec = ReturnsSpec(method=ret_method)
            df_processed = to_returns(df_processed, spec)

        # --- Display Results ---
        st.subheader("Transformed Data")
        
        col_stats1, col_stats2 = st.columns(2)
        col_stats1.info(f"Shape: {df_processed.shape}")
        col_stats2.info(f"Date Range: {df_processed.index.min().date()} to {df_processed.index.max().date()}")
        
        st.dataframe(df_processed.tail())
        
        # Plot Transformed
        title_prefix = "Returns" if calc_returns else "Prices"
        fig_proc = px.line(df_processed, title=f"{title_prefix} (Resampled: {rs_rule if do_resample else 'No'})")
        st.plotly_chart(fig_proc, use_container_width=True)
        
        # Correlation Matrix (Bonus)
        if len(symbols) > 1 and calc_returns:
            st.subheader("Correlation Matrix")
            corr = df_processed.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Asset Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)