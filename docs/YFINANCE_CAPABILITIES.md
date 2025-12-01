# YFinance Data Capabilities

This document outlines the data available through the `yfinance` library and how it's used in the trading agent.

## Overview

`yfinance` is a Python library that provides access to financial data from Yahoo Finance. It's a community-maintained library that scrapes data from Yahoo Finance's website.

## Available Data Types

### 1. Historical Market Data (OHLCV)

**Method**: `yf.download()` or `Ticker.history()`

**Data Available**:
- Open, High, Low, Close prices
- Adjusted Close prices
- Volume
- Dividends
- Stock splits

**Supported Intervals**:
- **Intraday**: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`
- **Hourly**: `1h`
- **Daily**: `1d`, `5d`
- **Weekly**: `1wk`
- **Monthly**: `1mo`, `3mo`

**Supported Periods** (for `Ticker.history()`):
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**Parameters**:
```python
yf.download(
    tickers="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    interval="1d",  # or "1m", "1h", "1wk", "1mo", etc.
    progress=False,  # Set to False to suppress progress bar
    # Note: 'show_errors' is NOT a valid parameter
)
```

### 2. Company Information

**Method**: `Ticker.info`

**Data Available**:
- Company name, sector, industry
- Market capitalization
- P/E ratio, EPS, beta
- 52-week high/low
- Dividend yield
- Number of employees
- Business summary
- Website, address
- And many more fields (100+ attributes)

**Example**:
```python
ticker = yf.Ticker("AAPL")
info = ticker.info
print(info['sector'], info['marketCap'], info['peRatio'])
```

### 3. Options Data

**Method**: `Ticker.option_chain(date)` or `Ticker.options`

**Data Available**:
- Option chains for specific expiration dates
- Calls and Puts
- Strike prices
- Bid/Ask prices
- Implied volatility
- Open interest
- Volume
- Greeks (delta, gamma, theta, vega)

**Example**:
```python
ticker = yf.Ticker("AAPL")
expirations = ticker.options  # List of expiration dates
chain = ticker.option_chain(expirations[0])
calls = chain.calls
puts = chain.puts
```

### 4. Financial Statements

**Methods**: 
- `Ticker.financials` - Income statement
- `Ticker.balance_sheet` - Balance sheet
- `Ticker.cashflow` - Cash flow statement
- `Ticker.quarterly_financials` - Quarterly income statement
- `Ticker.quarterly_balance_sheet` - Quarterly balance sheet
- `Ticker.quarterly_cashflow` - Quarterly cash flow

**Data Available**:
- Revenue, expenses, net income
- Assets, liabilities, equity
- Operating, investing, financing cash flows
- Historical quarterly and annual data

### 5. Dividends and Stock Splits

**Methods**:
- `Ticker.dividends` - Historical dividend payments
- `Ticker.splits` - Historical stock splits
- `Ticker.actions` - Combined dividends and splits

**Data Available**:
- Dividend payment dates and amounts
- Stock split dates and ratios

### 6. Analyst Recommendations

**Method**: `Ticker.recommendations`

**Data Available**:
- Analyst ratings (Buy, Hold, Sell, etc.)
- Recommendation dates
- Firm names

### 7. Institutional Holders

**Method**: `Ticker.institutional_holders`

**Data Available**:
- Institution names
- Shares held
- Percentage of shares
- Value of holdings
- Date reported

### 8. Major Holders

**Method**: `Ticker.major_holders`

**Data Available**:
- Top institutional holders
- Top mutual fund holders
- Insider ownership

### 9. Earnings Data

**Method**: `Ticker.earnings` or `Ticker.quarterly_earnings`

**Data Available**:
- Earnings per share (EPS)
- Revenue
- Historical quarterly/annual earnings

### 10. Calendar Events

**Method**: `Ticker.calendar`

**Data Available**:
- Earnings dates
- Ex-dividend dates
- Other important dates

### 11. Sustainability/ESG Scores

**Method**: `Ticker.sustainability`

**Data Available**:
- Environmental, Social, Governance scores
- Controversy level
- Various ESG metrics

### 12. News

**Method**: `Ticker.news`

**Data Available**:
- Recent news articles
- Headlines
- Publication dates
- Article links

## Current Implementation Status

### ✅ Implemented in YFinanceDataAdapter

1. **Historical Bars** (`get_bars()`)
   - Supports all intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
   - Returns OHLCV data as `Bar` objects
   - Handles date range filtering

2. **Option Chains** (`get_option_chain()`)
   - Fetches option chains for a given date
   - Finds closest expiration date
   - Returns `OptionContract` objects

### ❌ Not Yet Implemented

The following data types are available but not yet integrated:

1. Company information (`Ticker.info`)
2. Financial statements
3. Dividends and splits
4. Analyst recommendations
5. Institutional holders
6. Earnings data
7. Calendar events
8. ESG scores
9. News

## Limitations and Considerations

1. **Rate Limiting**: Yahoo Finance may rate limit requests. Be respectful with request frequency.

2. **Data Accuracy**: Data is scraped from Yahoo Finance and may have occasional inaccuracies or delays.

3. **Premium Features**: Some historical data (especially intraday) may require Yahoo Finance premium subscription.

4. **API Changes**: Yahoo Finance may change their website structure, which could break yfinance temporarily.

5. **No Real-time Streaming**: yfinance doesn't support real-time streaming. Use `Ticker.history(period="1d", interval="1m")` for recent intraday data.

6. **Timezone**: Data timestamps are typically in the exchange's local timezone.

## Usage Examples

### Fetch Historical Data
```python
import yfinance as yf

# Method 1: Using download()
data = yf.download("AAPL", start="2024-01-01", end="2024-12-31", interval="1d")

# Method 2: Using Ticker object
ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y", interval="1d")
```

### Get Company Info
```python
ticker = yf.Ticker("AAPL")
info = ticker.info
print(f"Market Cap: ${info['marketCap']:,}")
print(f"P/E Ratio: {info['trailingPE']}")
```

### Get Options
```python
ticker = yf.Ticker("AAPL")
expirations = ticker.options
chain = ticker.option_chain(expirations[0])
print(chain.calls.head())
```

## References

- [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance)
- [yfinance PyPI Page](https://pypi.org/project/yfinance/)
- [Yahoo Finance](https://finance.yahoo.com/)

