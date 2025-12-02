# LLM Investment Copilot

**Automated portfolio management for busy professionals who want market-beating returns with minimal time investment.**

## Value Proposition

Are you a busy professional who wants to outperform the S&P 500 but don't have hours to spend managing your portfolio? This LLM-powered investment copilot is designed for you.

### Who This Is For

- **Busy professionals** who lack time for active portfolio management
- **Higher risk tolerance** investors seeking above-market returns
- **5-minute daily commitment** - just enough time to review and rebalance
- **Tech-savvy individuals** who trust AI-driven decision making

### What It Does

- **Leverages latest LLM models** (GPT-4, GPT-5) to analyze market trends and make trading decisions
- **Automated backtesting** to validate strategies before deploying capital
- **Live trading execution** via moomoo/Futu API integration
- **Multiple trading strategies** including leveraged ETF volatility swing, LLM trend detection, and options wheel strategies
- **Professional-grade charting** with TradingView-style visualizations

### Expected Outcomes

- **Target**: Outperform S&P 500 buy & hold returns
- **Risk**: Higher volatility strategies (leveraged ETFs, options)
- **Time**: ~5 minutes per day for portfolio review and rebalancing
- **Automation**: LLM handles complex market analysis and signal generation

---

> **Note**: This is a framework that requires configuration. You'll need to:
> - Set up your API tokens (see Configuration section below)
> - Configure your trading strategies based on your risk tolerance
> - Connect to your broker (moomoo/Futu) for live trading

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

## How It Works

1. **LLM Analysis**: The system uses advanced language models to analyze market conditions, detect trends, and generate trading signals
2. **Strategy Execution**: Multiple proven strategies (leveraged ETF volatility swing, trend detection, options wheel) execute trades automatically
3. **Risk Management**: Built-in position sizing, stop-loss logic, and portfolio rebalancing
4. **Performance Tracking**: Real-time performance metrics, equity curves, and comparison to benchmarks

## Quick Start

1. **Set up your API tokens** (see Configuration section below)
2. **Run a backtest** to validate strategies:
   ```bash
   python scripts/run_backtest_leveraged_etf.py
   ```
3. **Review results** in interactive charts
4. **Deploy to live trading** when ready (requires broker API setup)

## Project Structure

- `core/` — Python framework code (data, execution, strategies, backtest, live)
- `scripts/` — CLI entry points for backtest, live mode, and performance tests
- `config/` — YAML configuration files for strategies and environment settings
- `mcp/` — MCP server scaffolding (TypeScript) for tool-based control

