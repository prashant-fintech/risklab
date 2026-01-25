"""
Unit tests for market data core functionality.
Tests for Post 01 - Market Data Core: Returns, Curves, Calendars (Foundation)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from risklab_core.market_data import (
    to_returns, 
    resample_prices, 
    align_assets, 
    winsorize, 
    handle_outliers
)
from risklab_core.contracts import (
    ReturnsSpec, 
    ReSampleSpec, 
    AlignSpec, 
    OutlierSpec
)


class TestReturnsTransformation:
    """Test suite for to_returns() function"""
    
    @pytest.fixture
    def sample_prices(self):
        """Sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = {
            'AAPL': [100.0, 102.0, 101.0, 105.0, 107.0],
            'MSFT': [200.0, 201.0, 203.0, 202.0, 206.0]
        }
        return pd.DataFrame(data, index=dates)
    
    def test_simple_returns(self, sample_prices):
        """Test simple return calculation"""
        returns = to_returns(sample_prices, ReturnsSpec(method="simple"))
        
        # Check shape (should lose one row due to pct_change)
        assert returns.shape == (4, 2)
        
        # Check first return for AAPL: (102-100)/100 = 0.02
        assert abs(returns.iloc[0]['AAPL'] - 0.02) < 1e-6
        
        # Check first return for MSFT: (201-200)/200 = 0.005
        assert abs(returns.iloc[0]['MSFT'] - 0.005) < 1e-6
    
    def test_log_returns(self, sample_prices):
        """Test log return calculation"""
        returns = to_returns(sample_prices, ReturnsSpec(method="log"))
        
        # Check shape
        assert returns.shape == (4, 2)
        
        # Check first log return for AAPL: ln(102/100) ≈ 0.0198
        expected_log_return = np.log(102.0/100.0)
        assert abs(returns.iloc[0]['AAPL'] - expected_log_return) < 1e-6
    
    def test_invalid_method_raises_error(self, sample_prices):
        """Test that invalid method raises ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            to_returns(sample_prices, ReturnsSpec(method="invalid"))
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        result = to_returns(empty_df)
        assert result.empty
        assert result.shape == (0, 0)
    
    def test_dropna_behavior(self, sample_prices):
        """Test dropna parameter behavior"""
        # Add NaN to test dropna
        prices_with_nan = sample_prices.copy()
        prices_with_nan.iloc[1, 0] = np.nan  # NaN in AAPL
        
        returns_dropna = to_returns(prices_with_nan, ReturnsSpec(dropna=True))
        returns_keep_na = to_returns(prices_with_nan, ReturnsSpec(dropna=False))
        
        # With dropna=True, should have fewer NaN rows
        assert returns_dropna.isna().sum().sum() <= returns_keep_na.isna().sum().sum()


class TestPriceResampling:
    """Test suite for resample_prices() function"""
    
    @pytest.fixture
    def daily_prices(self):
        """Daily price data for resampling tests"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = {
            'AAPL': range(100, 110),
            'MSFT': range(200, 210)
        }
        return pd.DataFrame(data, index=dates)
    
    def test_daily_resample_no_change(self, daily_prices):
        """Test that daily resampling returns original data"""
        result = resample_prices(daily_prices, ReSampleSpec(rule="D"))
        pd.testing.assert_frame_equal(result, daily_prices)
    
    def test_weekly_resample_last(self, daily_prices):
        """Test weekly resampling with last value"""
        result = resample_prices(daily_prices, ReSampleSpec(rule="W", how="last"))
        
        # Should have fewer rows (weekly instead of daily)
        assert len(result) < len(daily_prices)
        assert result.index.freq == 'W-SUN'  # Default weekly frequency
    
    def test_monthly_resample_mean(self, daily_prices):
        """Test monthly resampling with mean"""
        result = resample_prices(daily_prices, ReSampleSpec(rule="M", how="mean"))
        
        # Should aggregate to monthly
        assert len(result) == 1  # All data in same month
        
        # Check mean calculation
        expected_aapl_mean = daily_prices['AAPL'].mean()
        assert abs(result.iloc[0]['AAPL'] - expected_aapl_mean) < 1e-6
    
    def test_invalid_resample_method(self, daily_prices):
        """Test invalid resampling method raises ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            resample_prices(daily_prices, ReSampleSpec(how="invalid"))
    
    def test_empty_dataframe_resample(self):
        """Test resampling empty DataFrame"""
        empty_df = pd.DataFrame()
        result = resample_prices(empty_df)
        assert result.empty


class TestAssetAlignment:
    """Test suite for align_assets() function"""
    
    @pytest.fixture
    def misaligned_assets(self):
        """Create assets with different date ranges for alignment testing"""
        # Asset 1: 5 days
        dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
        asset1 = pd.DataFrame({'AAPL': range(100, 105)}, index=dates1)
        
        # Asset 2: 3 days, overlapping but different range  
        dates2 = pd.date_range('2023-01-03', periods=3, freq='D')
        asset2 = pd.DataFrame({'MSFT': range(200, 203)}, index=dates2)
        
        # Combine with missing dates
        combined = pd.concat([asset1, asset2], axis=1)
        return combined
    
    def test_inner_join_alignment(self, misaligned_assets):
        """Test inner join removes rows with any NaN"""
        result = align_assets(misaligned_assets, AlignSpec(join="inner"))
        
        # Inner join should only keep dates where both assets have data
        assert not result.isna().any().any()
        
        # Should be smaller than original
        assert len(result) <= len(misaligned_assets)
    
    def test_outer_join_with_forward_fill(self, misaligned_assets):
        """Test outer join with forward fill"""
        result = align_assets(misaligned_assets, AlignSpec(join="outer", fill_method="ffill"))
        
        # Should have same length as original (outer join)
        assert len(result) == len(misaligned_assets)
        
        # Forward fill should reduce NaNs
        original_nans = misaligned_assets.isna().sum().sum()
        result_nans = result.isna().sum().sum()
        assert result_nans <= original_nans
    
    def test_backward_fill(self, misaligned_assets):
        """Test backward fill method"""
        result = align_assets(misaligned_assets, AlignSpec(fill_method="bfill"))
        
        # Should reduce NaN count
        original_nans = misaligned_assets.isna().sum().sum()
        result_nans = result.isna().sum().sum()
        assert result_nans <= original_nans
    
    def test_no_fill_method(self, misaligned_assets):
        """Test alignment with no fill method"""
        result = align_assets(misaligned_assets, AlignSpec(join="outer", fill_method=None))
        
        # Should preserve original NaN structure (outer join with no fill)
        original_nans = misaligned_assets.isna().sum().sum()
        result_nans = result.isna().sum().sum()
        assert result_nans == original_nans
    
    def test_invalid_align_method(self, misaligned_assets):
        """Test invalid alignment method raises ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            align_assets(misaligned_assets, AlignSpec(join="invalid"))
    
    def test_invalid_fill_method(self, misaligned_assets):
        """Test invalid fill method raises ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            align_assets(misaligned_assets, AlignSpec(fill_method="invalid"))


class TestOutlierHandling:
    """Test suite for outlier handling functions"""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Create data with obvious outliers"""
        np.random.seed(42)  # For reproducible tests
        normal_data = np.random.normal(0, 1, 100)
        
        # Add extreme outliers
        outlier_data = normal_data.copy()
        outlier_data[0] = 10.0   # Extreme high
        outlier_data[-1] = -10.0  # Extreme low
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        return pd.DataFrame({'returns': outlier_data}, index=dates)
    
    def test_winsorize_clips_extremes(self, data_with_outliers):
        """Test winsorizing clips extreme values"""
        result = winsorize(data_with_outliers, lower_q=0.05, upper_q=0.95)
        
        # Extreme values should be clipped
        assert result['returns'].max() < data_with_outliers['returns'].max()
        assert result['returns'].min() > data_with_outliers['returns'].min()
        
        # Should preserve shape
        assert result.shape == data_with_outliers.shape
    
    def test_handle_outliers_winsorize_method(self, data_with_outliers):
        """Test handle_outliers with winsorize method"""
        spec = OutlierSpec(method="winsorize", lower_q=0.01, upper_q=0.99)
        result = handle_outliers(data_with_outliers, spec)
        
        # Should clip extremes
        assert result['returns'].max() < data_with_outliers['returns'].max()
        assert result['returns'].min() > data_with_outliers['returns'].min()
    
    def test_handle_outliers_clip_method(self, data_with_outliers):
        """Test handle_outliers with clip method"""
        spec = OutlierSpec(method="clip", clip_low=-5.0, clip_high=5.0)
        result = handle_outliers(data_with_outliers, spec)
        
        # Values should be clipped to specified bounds
        assert result['returns'].max() <= 5.0
        assert result['returns'].min() >= -5.0
    
    def test_handle_outliers_none_method(self, data_with_outliers):
        """Test handle_outliers with no method returns original data"""
        spec = OutlierSpec(method=None)
        result = handle_outliers(data_with_outliers, spec)
        
        pd.testing.assert_frame_equal(result, data_with_outliers)
    
    def test_handle_outliers_invalid_method(self, data_with_outliers):
        """Test invalid outlier method raises ValidationError"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            handle_outliers(data_with_outliers, OutlierSpec(method="invalid"))
    
    def test_empty_dataframe_outliers(self):
        """Test outlier handling on empty DataFrame"""
        empty_df = pd.DataFrame()
        result = winsorize(empty_df)
        assert result.empty


class TestAcceptanceCriteria:
    """Test specific acceptance criteria from Post 01"""
    
    def test_asset_alignment_produces_same_index(self):
        """AC: Given 2 assets with missing dates, alignment produces same index"""
        # Create two assets with different date ranges
        dates1 = pd.date_range('2023-01-01', periods=5, freq='D')
        dates2 = pd.date_range('2023-01-03', periods=5, freq='D')
        
        asset1_data = pd.DataFrame({'A': [100, 101, 102, 103, 104]}, index=dates1)
        asset2_data = pd.DataFrame({'B': [200, 201, 202, 203, 204]}, index=dates2)
        
        # Combine assets
        combined = pd.concat([asset1_data, asset2_data], axis=1)
        
        # Align with inner join
        aligned = align_assets(combined, AlignSpec(join="inner"))
        
        # All rows should have data for both assets (no NaN)
        assert not aligned.isna().any().any()
        
        # Index should be consistent
        assert isinstance(aligned.index, pd.DatetimeIndex)
    
    def test_returns_computed_correctly_simple_method(self):
        """AC: Returns computed correctly for simple method"""
        prices = pd.DataFrame({
            'STOCK': [100.0, 110.0, 99.0, 108.9]
        }, index=pd.date_range('2023-01-01', periods=4, freq='D'))
        
        returns = to_returns(prices, ReturnsSpec(method="simple"))
        
        # Manual calculation: (110-100)/100 = 0.1
        expected_return_1 = (110.0 - 100.0) / 100.0
        assert abs(returns.iloc[0]['STOCK'] - expected_return_1) < 1e-10
        
        # Manual calculation: (99-110)/110 ≈ -0.1
        expected_return_2 = (99.0 - 110.0) / 110.0
        assert abs(returns.iloc[1]['STOCK'] - expected_return_2) < 1e-10
    
    def test_returns_computed_correctly_log_method(self):
        """AC: Returns computed correctly for log method"""
        prices = pd.DataFrame({
            'STOCK': [100.0, 110.0, 99.0]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        returns = to_returns(prices, ReturnsSpec(method="log"))
        
        # Manual calculation: ln(110/100)
        expected_return_1 = np.log(110.0 / 100.0)
        assert abs(returns.iloc[0]['STOCK'] - expected_return_1) < 1e-10
        
        # Manual calculation: ln(99/110)
        expected_return_2 = np.log(99.0 / 110.0)
        assert abs(returns.iloc[1]['STOCK'] - expected_return_2) < 1e-10


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_end_to_end_pipeline(self):
        """Test complete market data processing pipeline"""
        # Create sample data with missing values and outliers
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        data = pd.DataFrame({
            'STOCK_A': [100, 102, np.nan, 105, 107, 106, 109, 111, 113, 115],
            'STOCK_B': [200, np.nan, 203, 205, 207, 209, 211, 213, 215, 1000]  # 1000 is outlier
        }, index=dates)
        
        # Step 1: Handle outliers
        cleaned = handle_outliers(data, OutlierSpec(method="winsorize", lower_q=0.1, upper_q=0.9))
        
        # Step 2: Align assets (fill missing values)
        aligned = align_assets(cleaned, AlignSpec(fill_method="ffill"))
        
        # Step 3: Convert to returns
        returns = to_returns(aligned, ReturnsSpec(method="simple"))
        
        # Step 4: Resample to weekly
        weekly_returns = resample_prices(returns, ReSampleSpec(rule="W", how="last"))
        
        # Verify pipeline worked
        assert not returns.empty
        assert not weekly_returns.empty
        assert len(weekly_returns) < len(returns)  # Weekly should have fewer observations
        
        # Outlier should be handled (no extreme values in returns)
        assert returns.abs().max().max() < 1.0  # No extreme return values