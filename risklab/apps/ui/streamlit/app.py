import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import sys
import os

# Add the packages to Python path for Streamlit Cloud deployment
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use the working path for Streamlit Cloud
core_src = os.path.join(current_dir, '..', '..', 'packages', 'risklab_core', 'src')
sys.path.insert(0, os.path.abspath(core_src))

# Import Core Logic
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

# Import Factor Computation
from risklab_core.factors import compute_drawdown, compute_factors
from risklab_core.contracts.factors import FactorConfig, FactorSpec

# Page Config
st.set_page_config(
    page_title="RiskLab Explorer",
    page_icon="lab",
    layout="wide"
)

st.title("RiskLab — Market Data & Risk Factor Pipeline")
st.markdown("""
This interface demonstrates the **RiskLab Core** pipeline:  
`Source` -> `Resample` -> `Outliers` -> `Returns` -> `Factors`
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
    tab_raw, tab_pipeline, tab_factors = st.tabs(["Raw Data", "Pipeline Config", "Factor Analysis"])
    
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
    
    with tab_factors:
        st.subheader("Risk Factor Computation")
        
        if not calc_returns:
            st.warning("⚠️ Factor computation requires returns data. Please enable 'Calculate Returns' in the Pipeline Config tab.")
        elif df_processed.empty:
            st.warning("⚠️ No processed data available. Please load data and configure the pipeline.")
        else:
            # Factor Configuration Sidebar
            st.sidebar.header("3. Factor Configuration")
            
            st.sidebar.subheader("Rolling Volatility")
            enable_vol = st.sidebar.checkbox("Enable Rolling Volatility", value=True)
            vol_window = st.sidebar.slider("Volatility Window", 5, 60, 20) if enable_vol else 20
            vol_min_periods = st.sidebar.slider("Min Periods (Vol)", 1, vol_window, vol_window//2) if enable_vol else None
            
            st.sidebar.subheader("Rolling Mean")
            enable_mean = st.sidebar.checkbox("Enable Rolling Mean", value=True)
            mean_window = st.sidebar.slider("Mean Window", 5, 60, 30) if enable_mean else 30
            
            st.sidebar.subheader("Drawdown")
            enable_dd = st.sidebar.checkbox("Enable Drawdown", value=True)
            
            st.sidebar.subheader("Rolling Correlation")
            enable_corr = st.sidebar.checkbox("Enable Rolling Correlation", value=len(symbols) > 1)
            corr_window = st.sidebar.slider("Correlation Window", 10, 90, 60) if enable_corr else 60
            corr_benchmark = st.sidebar.selectbox("Benchmark Asset", symbols, index=0) if enable_corr and len(symbols) > 1 else None
            
            # Build Factor Configuration
            factors = []
            
            if enable_vol:
                factors.append(FactorSpec(
                    name=f"vol_{vol_window}d",
                    kind="rolling_vol",
                    window=vol_window,
                    min_periods=vol_min_periods
                ))
            
            if enable_mean:
                factors.append(FactorSpec(
                    name=f"mean_{mean_window}d",
                    kind="rolling_mean",
                    window=mean_window
                ))
                
            if enable_dd:
                factors.append(FactorSpec(
                    name="drawdown",
                    kind="drawdown"
                ))
                
            if enable_corr and len(symbols) > 1 and corr_benchmark:
                factors.append(FactorSpec(
                    name=f"corr_{corr_benchmark.lower()}_{corr_window}d",
                    kind="rolling_corr",
                    window=corr_window,
                    benchmark=corr_benchmark
                ))
            
            if not factors:
                st.warning("⚠️ Please enable at least one factor type in the sidebar.")
            else:
                # Compute Factors
                with st.spinner("Computing risk factors..."):
                    try:
                        config = FactorConfig(factors=factors)
                        df_factors = compute_factors(df_processed, config)
                        
                        # Display Factor Results
                        st.success(f"✅ Computed {len(factors)} factor types across {len(symbols)} assets = {df_factors.shape[1]} total factors")
                        
                        # Factor Statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Factors", df_factors.shape[1])
                        col2.metric("Time Series Length", df_factors.shape[0])
                        col3.metric("Data Completeness", f"{(1 - df_factors.isna().sum().sum() / df_factors.size) * 100:.1f}%")
                        
                        # Factor Data Table
                        st.subheader("Factor Data (Latest 10 observations)")
                        st.dataframe(df_factors.tail(10), use_container_width=True)
                        
                        # Factor Visualizations
                        st.subheader("Factor Visualizations")
                        
                        # Create visualization tabs
                        viz_tabs = st.tabs(["Time Series", "Distributions", "Factor Correlation", "Heatmap"])
                        
                        with viz_tabs[0]:  # Time Series
                            # Allow user to select which factors to plot
                            factor_cols = list(df_factors.columns)
                            selected_factors = st.multiselect(
                                "Select Factors to Plot",
                                factor_cols,
                                default=factor_cols[:min(6, len(factor_cols))]  # Default to first 6
                            )
                            
                            if selected_factors:
                                fig_ts = px.line(
                                    df_factors[selected_factors],
                                    title="Factor Time Series",
                                    labels={"index": "Date", "value": "Factor Value"}
                                )
                                fig_ts.update_layout(hovermode='x unified')
                                st.plotly_chart(fig_ts, use_container_width=True)
                        
                        with viz_tabs[1]:  # Distributions
                            # Factor distributions
                            factor_for_dist = st.selectbox("Select Factor for Distribution", factor_cols)
                            if factor_for_dist:
                                factor_data = df_factors[factor_for_dist].dropna()
                                if not factor_data.empty:
                                    fig_hist = px.histogram(
                                        x=factor_data,
                                        title=f"Distribution of {factor_for_dist}",
                                        nbins=50,
                                        labels={"x": "Factor Value", "count": "Frequency"}
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                    
                                    # Summary statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Mean", f"{factor_data.mean():.4f}")
                                    col2.metric("Std Dev", f"{factor_data.std():.4f}")
                                    col3.metric("Skewness", f"{factor_data.skew():.4f}")
                                    col4.metric("Kurtosis", f"{factor_data.kurtosis():.4f}")
                        
                        with viz_tabs[2]:  # Factor Correlation
                            if len(factor_cols) > 1:
                                factor_corr = df_factors.corr()
                                fig_factor_corr = px.imshow(
                                    factor_corr,
                                    text_auto=True,
                                    title="Factor Correlation Matrix",
                                    color_continuous_scale="RdBu_r",
                                    aspect="auto"
                                )
                                fig_factor_corr.update_layout(height=600)
                                st.plotly_chart(fig_factor_corr, use_container_width=True)
                            else:
                                st.info("Need at least 2 factors for correlation analysis")
                        
                        with viz_tabs[3]:  # Heatmap
                            # Factor heatmap over time (recent data)
                            recent_factors = df_factors.tail(30)  # Last 30 observations
                            if not recent_factors.empty:
                                # Normalize for better visualization
                                factor_normalized = recent_factors.div(recent_factors.abs().max())
                                
                                fig_heatmap = px.imshow(
                                    factor_normalized.T,
                                    title="Factor Heatmap (Recent 30 periods, normalized)",
                                    labels={"x": "Date", "y": "Factor", "color": "Normalized Value"},
                                    color_continuous_scale="RdBu_r",
                                    aspect="auto"
                                )
                                fig_heatmap.update_layout(height=500)
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Download Section
                        st.subheader("Export Data")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download processed returns
                            csv_returns = df_processed.to_csv()
                            st.download_button(
                                label="📥 Download Returns Data (CSV)",
                                data=csv_returns,
                                file_name=f"risklab_returns_{date.today()}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Download factor data
                            csv_factors = df_factors.to_csv()
                            st.download_button(
                                label="📥 Download Factor Data (CSV)",
                                data=csv_factors,
                                file_name=f"risklab_factors_{date.today()}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error computing factors: {e}")
                        st.exception(e)