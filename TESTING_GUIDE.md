# Testing Guide - Revised MP2.0 Strategy

Revised MP2.0 is an improved version built on top of Mystic Pulse 2.0.

## Prerequisites

1. **Install dependencies** (if not already installed):
   ```bash
   pip install flask httpx PyYAML numpy
   ```

2. **Configure API Token**:
   - Edit `config/env.backtest.yaml`
   - Set your MarketData.app API token

3. **Configure Strategy**:
   - Edit `config/strategy.mystic_pulse.yaml`
   - Adjust parameters as needed:
     - `symbol`: Stock symbol (e.g., "AAPL", "TQQQ")
     - `timeframe`: "D" for daily or "W" for weekly
     - `min_trend_score`: Minimum score to trigger signals (default: 5)
     - `adx_length`: ADX smoothing period (default: 9)

## Running the Backtest

### Basic Test

```bash
python scripts/run_backtest_mystic_pulse.py
```

### What to Expect

1. **Backtest Execution**:
   - The script will load configuration files
   - Start the backtest engine
   - Process bars and generate signals
   - Display progress logs

2. **Web Server**:
   - After backtest completes, a Flask server starts on `http://localhost:5000`
   - Your browser should automatically open
   - If not, manually navigate to `http://localhost:5000`

3. **Web Interface**:
   - **Performance Metrics**: Total Return, CAGR, Sharpe Ratio, Max Drawdown
   - **Price Chart**: Shows price movement with BUY (green ▲) and SELL (red ▼) signals
   - **Volume Chart**: Trading volume bars
   - **Revised MP2.0 Indicator**: Positive/negative trend counts
   - **Equity Curve**: Portfolio value over time

4. **Stopping**:
   - Press `Enter` in the terminal to stop the server
   - Or press `Ctrl+C` to interrupt (will show partial results)

## Testing Different Configurations

### Test with Weekly Candles

Edit `config/strategy.mystic_pulse.yaml`:
```yaml
timeframe: W  # Weekly candles
```

### Test with Different Symbols

Edit `config/strategy.mystic_pulse.yaml`:
```yaml
symbol: TQQQ  # Or any other symbol
```

### Test with Different Sensitivity

Edit `config/strategy.mystic_pulse.yaml`:
```yaml
min_trend_score: 3  # Lower = more signals (more sensitive)
# or
min_trend_score: 10  # Higher = fewer signals (less sensitive)
```

## Troubleshooting

### Flask Not Found Error
```bash
pip install flask
```

### No Chart Displaying
- Check that the browser opened to `http://localhost:5000`
- Check terminal for any error messages
- Verify bars were retrieved (check log output)

### Port Already in Use
If port 5000 is busy, edit `scripts/run_backtest_mystic_pulse.py`:
```python
server = WebChartServer(port=5001)  # Use different port
```

### API Errors
- Verify your API token in `config/env.backtest.yaml`
- Check internet connection
- Verify the date range has available data

## Quick Test Script

For a quick test with minimal data:

1. Edit `config/env.backtest.yaml`:
   ```yaml
   start: "2024-11-01T06:30:00"  # Recent date
   end: "2024-11-30T13:00:00"    # Short range
   ```

2. Run:
   ```bash
   python scripts/run_backtest_mystic_pulse.py
   ```

This will run a shorter backtest for faster testing.

