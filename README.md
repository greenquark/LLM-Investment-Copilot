# Trading Agent Framework

This is a scaffold of a trading agent framework designed for:

- **Backtesting** using MarketData.app
- **Live trading** using moomoo / Futu API
- A **Wheel options strategy** driven by 15-minute decision intervals
- Integration with an MCP server for tool-based control

> NOTE: This is a skeleton project. You still need to:
> - Plug in your actual Wheel strategy canvas logic into `core/strategy/wheel.py`
> - Wire real moomoo / futu OpenAPI authentication and subscription details
> - Provide your own MarketData.app API token in `config/env.backtest.yaml`

## Structure

- `core/` — Python framework code (data, execution, strategies, backtest, live)
- `scripts/` — CLI entry points for backtest, live mode, and performance tests
- `config/` — YAML configuration files
- `mcp/` — MCP server scaffolding (TypeScript)

