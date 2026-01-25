# RiskLab 🧪

**A Modern Financial Risk Management Platform**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://risklab.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

RiskLab is a modular, plugin-based financial risk management platform built for modern quantitative finance. It provides a comprehensive suite of tools for market data processing, risk factor computation, VaR/ES calculation, stress testing, and portfolio analytics.

## 🌟 Features

### ✅ **Market Data Core (Post 01 - COMPLETE)**
- **Price Transformations**: Convert prices to simple/log returns with configurable methods
- **Asset Alignment**: Handle missing data across multiple assets with various join strategies  
- **Resampling**: Transform daily data to weekly/monthly frequencies
- **Outlier Handling**: Winsorize or clip extreme values for cleaner analysis
- **Business Calendars**: Extensible calendar system for holiday-aware calculations

### 🚀 **Interactive Demo**
Experience RiskLab's capabilities live: **[https://risklab.streamlit.app/](https://risklab.streamlit.app/)**

The demo showcases:
- Real-time market data fetching from Yahoo Finance
- Interactive price-to-returns transformation
- Asset alignment with different missing data policies
- Resampling functionality with visual comparisons
- Outlier detection and handling

### 🏗️ **Upcoming Features (In Development)**
- **Risk Factor Store**: Feature engineering layer for systematic risk factors
- **Volatility Models**: EWMA and GARCH forecasting engines  
- **VaR/ES Engines**: Parametric and historical Value-at-Risk calculations
- **Backtesting Framework**: Kupiec, Christoffersen, and Basel traffic-light validation
- **Stress Testing**: Scenario-driven shock engine with YAML DSL
- **Portfolio Analytics**: Risk parity and min-variance optimization
- **Interest Rate Risk**: Duration, convexity, and key rate sensitivities

## 🏛️ Architecture

RiskLab follows a **modular plugin architecture** with three core layers:

```
risklab/
├── packages/
│   ├── risklab_core/          # Central contracts, registry, and orchestration
│   ├── risklab_adapters/      # Infrastructure (DB, storage, external APIs)
│   └── risklab_plugins/       # Domain-specific risk modules
│       ├── market_risk/       # VaR, ES, stress testing
│       ├── credit_risk/       # PD, LGD, ECL models  
│       ├── op_risk/           # Operational risk (LDA)
│       ├── fraud_aml/         # Anomaly detection
│       └── limits_controls/   # Rules engine
├── apps/
│   ├── api/                   # FastAPI service layer
│   ├── ui/streamlit/          # Interactive dashboard
│   └── worker/                # Background processing
└── configs/                   # Platform and plugin configuration
```

### Key Design Principles
- **Contract-Driven**: All data flows use Pydantic contracts for validation
- **Plugin-Based**: New risk modules integrate via standardized interfaces
- **Service-Oriented**: API-first design with FastAPI backend
- **Configuration-Driven**: YAML-based configuration for domain rules
- **Test-Driven**: Comprehensive unit and integration test coverage

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **uv** package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prashant-fintech/risklab.git
   cd risklab
   ```

2. **Install with uv (recommended)**
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   
   cd risklab
   uv sync
   ```

3. **Or install with pip**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   
   cd risklab
   pip install -e .
   ```

### Running the Demo

**Local Streamlit App:**
```bash
cd risklab/apps/ui/streamlit
streamlit run app.py
```

**API Server:**
```bash
cd risklab/apps/api
uvicorn app.main:app --reload
```

**Run Tests:**
```bash
cd risklab
pytest tests/ -v
```

## 📊 Usage Examples

### Market Data Transformations

```python
from risklab_core.market_data import to_returns, align_assets, resample_prices
from risklab_core.contracts.market_data import ReturnsSpec, ReSampleSpec, AlignSpec

# Convert prices to returns
returns = to_returns(price_data, ReturnsSpec(method="log", dropna=True))

# Align multiple assets and handle missing data  
aligned = align_assets(multi_asset_data, AlignSpec(join="outer", fill_method="ffill"))

# Resample to weekly frequency
weekly = resample_prices(daily_data, ReSampleSpec(rule="W", how="last"))
```

### Outlier Handling

```python
from risklab_core.market_data.outliers import handle_outliers
from risklab_core.contracts.market_data import OutlierSpec

# Winsorize extreme values
clean_data = handle_outliers(
    return_data, 
    OutlierSpec(method="winsorize", lower_q=0.01, upper_q=0.99)
)

# Or clip to absolute bounds
clipped = handle_outliers(
    return_data,
    OutlierSpec(method="clip", clip_low=-0.1, clip_high=0.1)
)
```

## 🧪 Testing

RiskLab maintains comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/unit/test_market_data.py -v
pytest tests/unit/test_contracts.py -v

# Check test coverage
pytest --cov=risklab_core tests/
```

**Current Test Status**: ✅ 48 tests passing with comprehensive coverage of core functionality.

## 📁 Project Structure

```
risklab/
├── .github/
│   └── copilot-instructions.md    # AI development guidelines
├── notebooks/                     # Jupyter demonstrations
│   └── market_data_core.ipynb    # Post 01 validation notebook
├── backlog/                       # Feature specifications
│   └── first20.csv               # Detailed requirements
├── configs/                       # Configuration files
│   ├── platform.yaml            # Platform settings
│   └── plugins/                  # Plugin-specific configs
├── packages/                      # Core Python packages
│   ├── risklab_core/             # Central framework
│   ├── risklab_adapters/         # Infrastructure adapters
│   └── risklab_plugins/          # Risk domain modules
├── apps/                         # Service applications
│   ├── api/                      # FastAPI backend
│   ├── ui/streamlit/             # Interactive frontend
│   └── worker/                   # Background processor
└── tests/                        # Test suites
    ├── unit/                     # Unit tests
    └── integration/              # Integration tests
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`  
3. **Follow our development patterns**:
   - Use the plugin architecture for new risk modules
   - Write comprehensive tests for all functionality
   - Follow the contract-driven design patterns
   - Update documentation and examples

4. **Run the test suite**: `pytest tests/ -v`
5. **Submit a pull request**

### Development Workflow

RiskLab uses a **plugin-first development approach**:

1. **Define contracts** in `risklab_core/contracts/`
2. **Implement plugins** in `risklab_plugins/{domain}/`
3. **Add API endpoints** in `apps/api/routers/`
4. **Create tests** in `tests/unit/` and `tests/integration/`
5. **Update configurations** in `configs/plugins/`

## 🗺️ Roadmap

### Phase 1: Foundation (Q1 2026) ✅
- [x] Market Data Core
- [ ] Risk Factor Store  
- [ ] Volatility Models (EWMA/GARCH)

### Phase 2: Core Risk Engines (Q2 2026)
- [ ] VaR/ES Calculation Engines
- [ ] Backtesting Framework
- [ ] Stress Testing Library

### Phase 3: Advanced Analytics (Q3 2026)
- [ ] Portfolio Optimization
- [ ] Interest Rate Risk
- [ ] Credit Risk Models

### Phase 4: Production Features (Q4 2026)
- [ ] Operational Risk (LDA)
- [ ] Fraud/AML Detection
- [ ] Limits & Controls Engine

## 📈 Performance & Scalability

- **Vectorized Operations**: NumPy/Pandas for efficient numerical computation
- **Async Processing**: FastAPI with async/await for concurrent request handling
- **Modular Design**: Plugin architecture enables horizontal scaling
- **Caching**: Configurable result caching for expensive computations
- **Container Ready**: Docker configurations for cloud deployment

## 📖 Documentation

- **Live Demo**: [https://risklab.streamlit.app/](https://risklab.streamlit.app/)
- **API Documentation**: Available at `/docs` when running the FastAPI server
- **Notebooks**: Interactive examples in the `notebooks/` directory
- **Architecture Guide**: See `.github/copilot-instructions.md` for detailed patterns

## 🏷️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Project Maintainer**: Prashant  
**GitHub**: [@prashant-fintech](https://github.com/prashant-fintech)  
**Live Demo**: [https://risklab.streamlit.app/](https://risklab.streamlit.app/)

---

**RiskLab** - *Empowering Modern Risk Management with Open Source Innovation* 🚀
