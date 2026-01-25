# RiskLab Copilot Instructions

## Architecture Overview
RiskLab is a modular financial risk management platform built around a **plugin architecture** with three main layers:

### Core Architecture Pattern
- **`risklab_core/`**: Central kernel defining contracts, registry, and service orchestration
- **`risklab_adapters/`**: Infrastructure adapters (DB via SQLAlchemy, local file storage)
- **`risklab_plugins/`**: Domain-specific risk modules (market_risk, credit_risk, op_risk, fraud_aml, limits_controls)

### Service Architecture
- **API**: FastAPI service in `apps/api/` with routers for plugins, runs, and health
- **UI**: Streamlit dashboard in `apps/ui/streamlit/` 
- **Worker**: Background processing in `apps/worker/`

## Key Development Patterns

### Plugin Development
All risk plugins follow this structure:
```
packages/risklab_plugins/{domain}/src/risklab_{domain}/
├── __init__.py
├── plugin.py          # Main plugin interface
└── {specific_files}    # Domain logic (var.py, pd.py, lda.py, etc.)
```

**Plugin naming convention**: `risklab_{domain}` (e.g., `risklab_market_risk`, `risklab_credit_risk`)

### Configuration Management
- Platform config: `configs/platform.yaml`
- Plugin-specific config: `configs/plugins/{plugin}.yaml`
- All configs follow YAML format for domain rules and parameters

### Contracts & Registry System
- Base contracts in `packages/risklab_core/src/risklab_core/contracts/`
- Plugin registry for discovery: `packages/risklab_core/src/risklab_core/registry/`
- Run tracking system: `packages/risklab_core/src/risklab_core/runs/`

## Development Workflow

### Project Initialization
Run `init-risklab.bat` to scaffold complete project structure - this creates the full directory tree with placeholder files.

### Package Structure
Each package has standard Python structure:
```
packages/{package_name}/
├── pyproject.toml
└── src/{package_name}/
    ├── __init__.py
    └── {modules}
```

### API Integration Pattern
FastAPI routers in `apps/api/app/routers/` correspond to plugin domains:
- `plugins.py`: Plugin registry and discovery
- `runs.py`: Execution and audit trail
- `health.py`: System health checks

### Testing Strategy
- Unit tests: `tests/unit/test_{component}.py`
- Integration tests: `tests/integration/test_{service}.py`
- Focus on contracts and registry functionality

## Domain-Specific Conventions

### Risk Module Patterns
Based on the backlog, expect these module types:
- **Market Risk**: VaR/ES engines, backtesting, stress testing (`var.py`, `es.py`, `backtest.py`, `stress.py`)
- **Credit Risk**: PD/LGD/EAD models, ECL calculation (`pd.py`, `lgd.py`, `ead.py`, `ecl.py`)
- **Operational Risk**: Loss Distribution Approach (`lda.py`)
- **Fraud/AML**: Anomaly detection (`anomaly.py`)
- **Limits & Controls**: Rules engine (`rules.py`)

### Data Flow Pattern
1. **Input**: Through API endpoints or config files
2. **Processing**: Via plugin-specific modules
3. **Storage**: Run metadata and results in DB (via adapters)
4. **Output**: JSON responses, dashboard visualizations

## Key Integration Points

### Dependency Injection
Uses container pattern in `apps/api/app/deps/container.py` for service wiring.

### Request Flow
1. API router → plugin registry → specific plugin → adapter (storage) → response
2. All runs tracked with `run_id` for audit trails
3. Middleware in `apps/api/app/middleware/request_id.py` for correlation

### Error Handling
Central error definitions in `packages/risklab_core/src/risklab_core/errors.py`

## Essential Files to Reference
- [`init-risklab.bat`](init-risklab.bat): Complete project structure blueprint
- [`backlog/first20.csv`](backlog/first20.csv): Detailed feature specifications and module relationships
- Plugin configuration pattern: [`configs/plugins/`](risklab/configs/plugins/)
- Core contracts: [`packages/risklab_core/src/risklab_core/contracts/`](risklab/packages/risklab_core/src/risklab_core/contracts/)

## Development Commands
- Project setup: `init-risklab.bat` (generates full structure)
- Service orchestration: Docker Compose (configurations in `apps/*/Dockerfile`)
- Testing: Standard pytest pattern for each package

When implementing new features, always consider the plugin architecture and ensure proper integration with the registry system and run tracking.