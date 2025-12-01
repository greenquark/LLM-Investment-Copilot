# LLM Investment Copilot

This is an LLM-powered investment copilot framework designed for:

- **Backtesting** using MarketData.app
- **Live trading** using moomoo / Futu API
- A **Wheel options strategy** driven by 15-minute decision intervals
- Integration with an MCP server for tool-based control

> NOTE: This is a skeleton project. You still need to:
> - Plug in your actual Wheel strategy canvas logic into `core/strategy/wheel.py`
> - Wire real moomoo / futu OpenAPI authentication and subscription details
> - Set up your API tokens (see Configuration section below)

## Configuration

### API Tokens and Secrets

**Important**: Never commit API tokens to version control. This project uses a separate `secrets.yaml` file that is gitignored.

1. **Copy the secrets template**:
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   ```

2. **Fill in your API tokens** in `config/secrets.yaml`:
   - `data_sources.marketdata_app.api_token` - Your MarketData.app API token
   - `strategies.llm_trend_detection.openai_api_key` - Your OpenAI API key
   - `moomoo.account_id` - Your Moomoo account ID (for live trading)

3. The `config/secrets.yaml` file is automatically gitignored and will never be committed.

4. Config files (`config/env.*.yaml` and `config/strategy.*.yaml`) contain placeholders and will automatically load secrets from `config/secrets.yaml` when available.

## Creating Custom Strategies

To create your own trading strategy:

1. **Copy the example skeleton**:
   ```bash
   cp core/strategy/example_strategy.py core/strategy/my_strategy.py
   cp config/strategy.example.yaml config/strategy.my_strategy.yaml
   ```

2. **Rename the class** in `my_strategy.py`:
   - Rename `ExampleStrategy` to `MyStrategy`
   - Rename `ExampleStrategyConfig` to `MyStrategyConfig`
   - Update the `from_dict` method if needed

3. **Implement your strategy logic**:
   - `on_start()`: Initialize indicators, load historical data
   - `on_decision()`: Main trading logic - fetch data, calculate signals, execute trades
   - `on_end()`: Cleanup, close positions, generate reports

4. **Update .gitignore** to exclude your implementation:
   ```
   core/strategy/my_strategy.py
   ```

5. **Create a backtest script** (see `scripts/run_backtest_*.py` for examples)

See `core/strategy/example_strategy.py` for detailed documentation and code structure.

## Structure

- `core/` — Python framework code (data, execution, strategies, backtest, live)
- `scripts/` — CLI entry points for backtest, live mode, and performance tests
- `config/` — YAML configuration files
- `mcp/` — MCP server scaffolding (TypeScript)

