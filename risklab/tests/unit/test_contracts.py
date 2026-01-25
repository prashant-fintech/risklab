"""
Unit tests for RiskLab contracts and data models.
Tests for data contracts used in market data processing.
"""
import pytest
from datetime import date
from pydantic import ValidationError

from risklab_core.contracts import (
    PriceRequest,
    PriceResponse, 
    ReturnsSpec,
    ReSampleSpec,
    AlignSpec,
    OutlierSpec,
    VendorMeta
)


class TestPriceRequest:
    """Test PriceRequest contract validation"""
    
    def test_valid_price_request(self):
        """Test valid price request creation"""
        request = PriceRequest(
            symbols=["AAPL", "MSFT"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            field="adj_close"
        )
        
        assert request.symbols == ["AAPL", "MSFT"]
        assert request.field == "adj_close"
        assert request.tz == "UTC"  # Default value
    
    def test_invalid_field_raises_error(self):
        """Test that invalid field raises validation error"""
        with pytest.raises(ValidationError):
            PriceRequest(
                symbols=["AAPL"],
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                field="invalid_field"
            )
    
    def test_empty_symbols_list(self):
        """Test handling of empty symbols list"""
        request = PriceRequest(
            symbols=[],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        assert request.symbols == []


class TestReturnsSpec:
    """Test ReturnsSpec contract validation"""
    
    def test_default_returns_spec(self):
        """Test default values"""
        spec = ReturnsSpec()
        assert spec.method == "simple"
        assert spec.dropna == True
    
    def test_valid_methods(self):
        """Test valid return methods"""
        simple_spec = ReturnsSpec(method="simple")
        log_spec = ReturnsSpec(method="log")
        
        assert simple_spec.method == "simple"
        assert log_spec.method == "log"
    
    def test_invalid_method_raises_error(self):
        """Test invalid method raises validation error"""
        with pytest.raises(ValidationError):
            ReturnsSpec(method="invalid")


class TestReSampleSpec:
    """Test ReSampleSpec contract validation"""
    
    def test_default_resample_spec(self):
        """Test default values"""
        spec = ReSampleSpec()
        assert spec.rule == "D"
        assert spec.how == "last"
        assert spec.label == "right"
        assert spec.closed == "right"
    
    def test_valid_frequency_rules(self):
        """Test valid frequency rules"""
        for rule in ["D", "W", "M", "Q", "Y"]:
            spec = ReSampleSpec(rule=rule)
            assert spec.rule == rule
    
    def test_valid_aggregation_methods(self):
        """Test valid aggregation methods"""
        for method in ["mean", "sum", "first", "last"]:
            spec = ReSampleSpec(how=method)
            assert spec.how == method
    
    def test_invalid_rule_raises_error(self):
        """Test invalid rule raises validation error"""
        with pytest.raises(ValidationError):
            ReSampleSpec(rule="invalid")


class TestAlignSpec:
    """Test AlignSpec contract validation"""
    
    def test_default_align_spec(self):
        """Test default values"""
        spec = AlignSpec()
        assert spec.join == "inner"
        assert spec.fill_method is None
    
    def test_valid_join_methods(self):
        """Test valid join methods"""
        for join_method in ["inner", "outer", "left", "right"]:
            spec = AlignSpec(join=join_method)
            assert spec.join == join_method
    
    def test_valid_fill_methods(self):
        """Test valid fill methods"""
        for fill_method in ["ffill", "bfill", "pad", "backfill", None]:
            spec = AlignSpec(fill_method=fill_method)
            assert spec.fill_method == fill_method
    
    def test_invalid_join_raises_error(self):
        """Test invalid join method raises validation error"""
        with pytest.raises(ValidationError):
            AlignSpec(join="invalid")


class TestOutlierSpec:
    """Test OutlierSpec contract validation"""
    
    def test_default_outlier_spec(self):
        """Test default values"""
        spec = OutlierSpec()
        assert spec.method is None
        assert spec.lower_q == 0.01
        assert spec.upper_q == 0.99
    
    def test_valid_methods(self):
        """Test valid outlier methods"""
        for method in [None, "winsorize", "clip"]:
            spec = OutlierSpec(method=method)
            assert spec.method == method
    
    def test_quantile_validation(self):
        """Test quantile parameter validation"""
        # Valid quantiles
        spec = OutlierSpec(lower_q=0.05, upper_q=0.95)
        assert spec.lower_q == 0.05
        assert spec.upper_q == 0.95
    
    def test_invalid_quantiles_raise_error(self):
        """Test invalid quantiles raise validation errors"""
        # lower_q too high
        with pytest.raises(ValidationError):
            OutlierSpec(lower_q=0.8)  # > 0.5
        
        # upper_q too low  
        with pytest.raises(ValidationError):
            OutlierSpec(upper_q=0.3)  # < 0.5
        
        # Negative quantiles
        with pytest.raises(ValidationError):
            OutlierSpec(lower_q=-0.1)
    
    def test_invalid_method_raises_error(self):
        """Test invalid method raises validation error"""
        with pytest.raises(ValidationError):
            OutlierSpec(method="invalid")


class TestVendorMeta:
    """Test VendorMeta contract"""
    
    def test_vendor_meta_creation(self):
        """Test creating VendorMeta object"""
        vendor = VendorMeta(
            vendor="YahooFinance",
            dataset="daily_prices"
        )
        
        assert vendor.vendor == "YahooFinance"
        assert vendor.dataset == "daily_prices"


class TestContractIntegration:
    """Integration tests for contract usage"""
    
    def test_contracts_work_together(self):
        """Test that contracts can be used together in typical workflow"""
        # Create request
        request = PriceRequest(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            field="adj_close",
            vendor="yahoo"
        )
        
        # Create processing specs
        returns_spec = ReturnsSpec(method="log", dropna=True)
        resample_spec = ReSampleSpec(rule="W", how="last")
        align_spec = AlignSpec(join="inner", fill_method="ffill")
        outlier_spec = OutlierSpec(method="winsorize", lower_q=0.02, upper_q=0.98)
        
        # Verify all specs are valid
        assert request.symbols == ["AAPL", "MSFT", "GOOGL"]
        assert returns_spec.method == "log"
        assert resample_spec.rule == "W"
        assert align_spec.join == "inner"
        assert outlier_spec.method == "winsorize"
    
    def test_contract_serialization(self):
        """Test contract JSON serialization/deserialization"""
        spec = ReturnsSpec(method="log", dropna=False)
        
        # Serialize to dict
        spec_dict = spec.model_dump()
        assert spec_dict["method"] == "log"
        assert spec_dict["dropna"] == False
        
        # Deserialize from dict
        spec_restored = ReturnsSpec(**spec_dict)
        assert spec_restored.method == "log"
        assert spec_restored.dropna == False