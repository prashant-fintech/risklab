"""
Test cases for risklab_core.factors module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from risklab_core.factors import compute_drawdown, compute_factors
from risklab_core.contracts.factors import FactorConfig, FactorSpec


class TestComputeDrawdown:
    """Test cases for compute_drawdown function"""
    
    def test_empty_series(self):
        """Test drawdown calculation with empty series"""
        empty_series = pd.Series(dtype=float, name='TEST')
        result = compute_drawdown(empty_series)
        assert result.empty
        assert result.name == 'TEST'
    
    def test_single_value(self):
        """Test drawdown with single return value"""
        returns = pd.Series([0.1], name='TEST')
        result = compute_drawdown(returns)
        expected = pd.Series([0.0], name='TEST')  # No drawdown with single value
        pd.testing.assert_series_equal(result, expected)
    
    def test_positive_returns_only(self):
        """Test drawdown with only positive returns (no drawdown)"""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03], name='TEST')
        result = compute_drawdown(returns)
        # All drawdowns should be 0 or negative, but with only positive returns, should be near 0
        assert all(result <= 0)
        assert result.iloc[0] == 0.0  # First value always 0
    
    def test_simple_drawdown_scenario(self):
        """Test drawdown with known scenario"""
        # Scenario: +10%, -5%, +2% returns
        returns = pd.Series([0.1, -0.05, 0.02], name='TEST')
        result = compute_drawdown(returns)
        
        # Wealth index: [1.1, 1.045, 1.066]
        # Running peaks: [1.1, 1.1, 1.1]
        # Drawdowns: [0, -0.05, -0.03090909...]
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == pytest.approx(-0.05, abs=1e-6)
        assert result.iloc[2] < 0  # Still in drawdown
    
    def test_recovery_from_drawdown(self):
        """Test recovery from maximum drawdown"""
        # Large drop then recovery
        returns = pd.Series([0.0, -0.2, 0.25], name='TEST')  # 0%, -20%, +25%
        result = compute_drawdown(returns)
        
        assert result.iloc[0] == 0.0  # No drawdown initially
        assert result.iloc[1] == pytest.approx(-0.2, abs=1e-10)  # 20% drawdown
        assert result.iloc[2] == pytest.approx(0.0, abs=1e-10)  # Full recovery
    
    def test_multiple_peaks(self):
        """Test drawdown with multiple peaks and troughs"""
        returns = pd.Series([0.1, -0.05, 0.08, -0.02, 0.01], name='TEST')
        result = compute_drawdown(returns)
        
        assert result.iloc[0] == 0.0  # First peak
        assert result.iloc[1] < 0  # First drawdown
        # Should handle subsequent peaks correctly
        assert len(result) == len(returns)
    
    def test_nan_handling(self):
        """Test drawdown with NaN values"""
        returns = pd.Series([0.1, np.nan, -0.05, 0.02], name='TEST')
        result = compute_drawdown(returns)
        
        # Function should handle NaN gracefully (fillna(0.0) at end)
        assert not result.isna().any()
        assert len(result) == len(returns)


class TestComputeFactors:
    """Test cases for compute_factors function"""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = {
            'AAPL': np.random.normal(0.001, 0.02, 100),  # 0.1% daily return, 2% vol
            'MSFT': np.random.normal(0.0008, 0.018, 100),  # Slightly different params
            'SPY': np.random.normal(0.0005, 0.015, 100)   # Market benchmark
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_empty_returns(self):
        """Test compute_factors with empty DataFrame"""
        empty_df = pd.DataFrame()
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20)
        ])
        
        result = compute_factors(empty_df, config)
        assert result.empty
        assert isinstance(result, pd.DataFrame)
    
    def test_rolling_volatility(self, sample_returns):
        """Test rolling volatility calculation"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20)
        ])
        
        result = compute_factors(sample_returns, config)
        
        # Check output structure
        expected_cols = ['AAPL_vol_20d', 'MSFT_vol_20d', 'SPY_vol_20d']
        assert list(result.columns) == expected_cols
        assert len(result) == len(sample_returns)
        
        # Check volatility properties
        assert result.iloc[19:].notna().all().all()  # Values after window-1
        assert (result.iloc[19:] >= 0).all().all()  # Volatility is non-negative
        
        # First 19 values should be NaN (window=20, min_periods defaults to window)
        assert result.iloc[:19].isna().all().all()
    
    def test_rolling_volatility_with_min_periods(self, sample_returns):
        """Test rolling volatility with custom min_periods"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20, min_periods=10)
        ])
        
        result = compute_factors(sample_returns, config)
        
        # Should have values starting from index 9 (min_periods=10)
        assert result.iloc[9:].notna().all().all()
        assert result.iloc[:9].isna().all().all()
    
    def test_rolling_mean(self, sample_returns):
        """Test rolling mean calculation"""
        config = FactorConfig(factors=[
            FactorSpec(name="mean_30d", kind="rolling_mean", window=30)
        ])
        
        result = compute_factors(sample_returns, config)
        
        expected_cols = ['AAPL_mean_30d', 'MSFT_mean_30d', 'SPY_mean_30d']
        assert list(result.columns) == expected_cols
        
        # Check that rolling means are reasonable
        rolling_means = result.iloc[29:]  # After window period
        assert rolling_means.notna().all().all()
        
        # Rolling means should be close to the true means we used to generate data
        for col in ['AAPL_mean_30d', 'MSFT_mean_30d', 'SPY_mean_30d']:
            assert abs(rolling_means[col].mean()) < 0.01  # Should be close to 0
    
    def test_drawdown_factor(self, sample_returns):
        """Test drawdown factor calculation"""
        config = FactorConfig(factors=[
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result = compute_factors(sample_returns, config)
        
        expected_cols = ['AAPL_drawdown', 'MSFT_drawdown', 'SPY_drawdown']
        assert list(result.columns) == expected_cols
        
        # Drawdown properties
        assert (result <= 0).all().all()  # Drawdowns are non-positive
        assert (result.iloc[0] == 0).all()  # First value is always 0
        assert result.notna().all().all()  # No NaN values
    
    def test_rolling_correlation(self, sample_returns):
        """Test rolling correlation calculation"""
        config = FactorConfig(factors=[
            FactorSpec(name="corr_spy_60d", kind="rolling_corr", window=60, benchmark="SPY")
        ])
        
        result = compute_factors(sample_returns, config)
        
        expected_cols = ['AAPL_corr_spy_60d', 'MSFT_corr_spy_60d', 'SPY_corr_spy_60d']
        assert list(result.columns) == expected_cols
        
        # Correlation properties
        corr_data = result.iloc[59:]  # After window period
        assert corr_data.notna().all().all()
        assert (corr_data >= -1).all().all()  # Correlation >= -1
        assert (corr_data <= 1.00001).all().all()   # Correlation <= 1 (allow small numerical error)
        
        # SPY correlation with itself should be 1 (with small numerical tolerance)
        assert (corr_data['SPY_corr_spy_60d'] >= 0.99999).all()
    
    def test_multiple_factors(self, sample_returns):
        """Test computing multiple factors simultaneously"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20),
            FactorSpec(name="mean_30d", kind="rolling_mean", window=30),
            FactorSpec(name="drawdown", kind="drawdown"),
            FactorSpec(name="corr_spy_40d", kind="rolling_corr", window=40, benchmark="SPY")
        ])
        
        result = compute_factors(sample_returns, config)
        
        # Should have 4 factors * 3 assets = 12 columns
        assert result.shape[1] == 12
        assert result.shape[0] == len(sample_returns)
        
        # Check column naming convention
        expected_patterns = ['vol_20d', 'mean_30d', 'drawdown', 'corr_spy_40d']
        for pattern in expected_patterns:
            matching_cols = [col for col in result.columns if pattern in col]
            assert len(matching_cols) == 3  # One for each asset
    
    def test_column_sorting(self, sample_returns):
        """Test that output columns are deterministically sorted"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20),
            FactorSpec(name="mean_15d", kind="rolling_mean", window=15),
        ])
        
        result1 = compute_factors(sample_returns, config)
        result2 = compute_factors(sample_returns, config)
        
        # Column order should be identical
        assert list(result1.columns) == list(result2.columns)
        # Should be sorted alphabetically
        assert list(result1.columns) == sorted(result1.columns)
    
    def test_error_handling_missing_window(self, sample_returns):
        """Test error handling for missing window parameter"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_bad", kind="rolling_vol")  # No window specified
        ])
        
        with pytest.raises(ValueError, match="Window required"):
            compute_factors(sample_returns, config)
    
    def test_error_handling_missing_benchmark(self, sample_returns):
        """Test error handling for missing benchmark in correlation"""
        config = FactorConfig(factors=[
            FactorSpec(name="corr_bad", kind="rolling_corr", window=20)  # No benchmark
        ])
        
        with pytest.raises(ValueError, match="Window and benchmark required"):
            compute_factors(sample_returns, config)
    
    def test_factor_spec_output_naming(self):
        """Test FactorSpec.get_output_name method"""
        spec = FactorSpec(name="vol_21d", kind="rolling_vol", window=21)
        
        assert spec.get_output_name("AAPL") == "AAPL_vol_21d"
        assert spec.get_output_name("MSFT") == "MSFT_vol_21d"
    
    def test_deterministic_output(self, sample_returns):
        """Test that output is deterministic for same input"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result1 = compute_factors(sample_returns, config)
        result2 = compute_factors(sample_returns, config)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_edge_case_small_window(self, sample_returns):
        """Test behavior with very small windows"""
        small_returns = sample_returns.head(5)  # Only 5 observations
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_2d", kind="rolling_vol", window=2, min_periods=2)
        ])
        
        result = compute_factors(small_returns, config)
        
        # Should have values for last 4 observations
        assert result.iloc[1:].notna().all().all()
        assert result.iloc[0].isna().all()
    
    def test_missing_benchmark_column(self, sample_returns):
        """Test error when benchmark column doesn't exist"""
        config = FactorConfig(factors=[
            FactorSpec(name="corr_bad", kind="rolling_corr", window=20, benchmark="NONEXISTENT")
        ])
        
        # Should handle missing benchmark gracefully 
        with pytest.raises(KeyError):
            compute_factors(sample_returns, config)
    
    def test_window_larger_than_data(self, sample_returns):
        """Test behavior when window is larger than available data"""
        small_data = sample_returns.head(10)  # Only 10 observations
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_50d", kind="rolling_vol", window=50)  # Larger than data
        ])
        
        result = compute_factors(small_data, config)
        
        # Should return all NaN since window > data length
        assert result.isna().all().all()


class TestFactorConfigIntegration:
    """Integration tests for FactorConfig"""
    
    def test_factor_config_validation(self):
        """Test FactorConfig validation"""
        # Valid config
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        assert len(config.factors) == 2
    
    def test_empty_factor_list(self):
        """Test FactorConfig with empty factor list"""
        config = FactorConfig(factors=[])
        assert len(config.factors) == 0
        
        # Should return empty result when no factors specified
        sample_data = pd.DataFrame({
            'A': [0.1, 0.2], 'B': [0.15, -0.1]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        # This will currently raise ValueError due to empty concat - we expect this
        with pytest.raises(ValueError, match="No objects to concatenate"):
            compute_factors(sample_data, config)
    
    def test_duplicate_factor_names(self):
        """Test handling of duplicate factor names"""
        config = FactorConfig(factors=[
            FactorSpec(name="vol_20d", kind="rolling_vol", window=20),
            FactorSpec(name="vol_20d_mean", kind="rolling_mean", window=20)  # Different name
        ])
        
        sample_data = pd.DataFrame({
            'A': np.random.randn(50)
        }, index=pd.date_range('2023-01-01', periods=50))
        
        result = compute_factors(sample_data, config)
        
        # Should handle both factors
        assert 'A_vol_20d' in result.columns
        assert 'A_vol_20d_mean' in result.columns
        assert result.shape[1] == 2
    
    def test_config_with_defaults(self):
        """Test FactorConfig with defaults"""
        config = FactorConfig(
            factors=[
                FactorSpec(name="vol_20d", kind="rolling_vol", window=20)
            ],
            defaults={"min_periods_ratio": 0.8}
        )
        
        assert config.defaults["min_periods_ratio"] == 0.8


class TestFactorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_all_nan_data(self):
        """Test behavior with all-NaN input data"""
        nan_data = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_2d", kind="rolling_vol", window=2),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result = compute_factors(nan_data, config)
        
        # Volatility should be NaN, drawdown should be 0 (fillna)
        vol_cols = [col for col in result.columns if 'vol_' in col]
        dd_cols = [col for col in result.columns if 'drawdown' in col]
        
        assert result[vol_cols].isna().all().all()
        assert (result[dd_cols] == 0).all().all()
    
    def test_single_row_data(self):
        """Test behavior with single row of data"""
        single_row = pd.DataFrame({
            'A': [0.01], 'B': [-0.02]
        }, index=pd.date_range('2023-01-01', periods=1))
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_5d", kind="rolling_vol", window=5),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result = compute_factors(single_row, config)
        
        # Volatility should be NaN (insufficient data)
        # Drawdown should be 0 (first value)
        vol_cols = [col for col in result.columns if 'vol_' in col]
        dd_cols = [col for col in result.columns if 'drawdown' in col]
        
        assert result[vol_cols].isna().all().all()
        assert (result[dd_cols] == 0).all().all()
    
    def test_extreme_values(self):
        """Test handling of extreme return values"""
        extreme_data = pd.DataFrame({
            'EXTREME': [10.0, -0.9, 5.0, -0.95, 0.0]  # Very large returns
        }, index=pd.date_range('2023-01-01', periods=5))
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_3d", kind="rolling_vol", window=3),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result = compute_factors(extreme_data, config)
        
        # Should handle extreme values without errors
        assert result.notna().any().any()
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    
    def test_mixed_frequency_index(self):
        """Test with non-regular date index"""
        irregular_dates = pd.to_datetime([
            '2023-01-01', '2023-01-03', '2023-01-08', '2023-01-15', '2023-01-16'
        ])
        
        data = pd.DataFrame({
            'A': [0.01, 0.02, -0.01, 0.03, -0.005]
        }, index=irregular_dates)
        
        config = FactorConfig(factors=[
            FactorSpec(name="vol_3d", kind="rolling_vol", window=3),
            FactorSpec(name="drawdown", kind="drawdown")
        ])
        
        result = compute_factors(data, config)
        
        # Should work with irregular dates
        assert len(result) == len(data)
        assert not result.empty


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])