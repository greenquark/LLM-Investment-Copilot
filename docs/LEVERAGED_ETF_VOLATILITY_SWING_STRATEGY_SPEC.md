# Leveraged ETF Volatility Swing Trading Strategy - Specification

**Note**: This strategy is NOT limited to SOXL/SOXS. It works with any bull/bear leveraged ETF pair.
Examples: SOXL/SOXS, TQQQ/SQQQ, UPRO/SPXU, TNA/TZA, LABU/LABD

## Document Version
- **Version**: 1.0
- **Date**: 2025-11-27
- **Source**: Leveraged ETF Swing Trade Strategy PDF + TypeScript Reference Implementation

---

## 1. Strategy Overview

### 1.1 Purpose
The SOXL/SOXS Volatility Swing Trading Strategy exploits daily and weekly price fluctuations in leveraged semiconductor ETFs using:
- **Trend filters** (regime detection via SOXX)
- **Volatility-based mean reversion**
- **Probability-driven targets**
- **Laddered execution**

### 1.2 Trading Instruments
This strategy works with any bull/bear leveraged ETF pair. Examples:
- **SOXL/SOXS**: Semiconductors (regime: SOXX or SMH)
- **TQQQ/SQQQ**: NASDAQ (regime: QQQ)
- **UPRO/SPXU**: S&P 500 (regime: SPY)
- **TNA/TZA**: Small caps (regime: IWM)
- **LABU/LABD**: Biotech (regime: XBI)

Each pair consists of:
- **Bull ETF**: 3x long leveraged ETF (e.g., SOXL, TQQQ, UPRO, TNA, LABU)
- **Bear ETF**: 3x short leveraged ETF (e.g., SOXS, SQQQ, SPXU, TZA, LABD)
- **Regime Symbol**: Unlevered underlying index ETF (e.g., SOXX, QQQ, SPY, IWM, XBI)

### 1.3 Key Constraints
- Trade only the configured bull/bear ETF pair (e.g., SOXL/SOXS, TQQQ/SQQQ, UPRO/SPXU)
- Typical holding time: **1-5 days** (swing trades)
- Maximum per-trade risk: **0.5-1.0%** of account equity
- **Never hold bull and bear ETFs simultaneously** in the same direction
- Trade frequency expectation: **2-5 trades per week**

---

## 2. Regime Filter (Daily Trend Direction)

### 2.1 Purpose
Use unlevered underlying index (e.g., SOXX, QQQ, SPY, IWM, XBI) to determine which instrument to trade and position sizing.

### 2.2 Regime Detection Logic

#### Bull Regime → Trade Long Bull ETF Only
**Conditions:**
- Price > EMA50
- EMA20 > EMA50

**Trading Rules:**
- Only enter long positions in bull ETF (e.g., SOXL, TQQQ, UPRO, TNA, LABU)
- Do not enter bear ETF positions (except rare counter-trend trades)

#### Bear Regime → Trade Long Bear ETF Only
**Conditions:**
- Price < EMA50
- EMA20 < EMA50

**Trading Rules:**
- Only enter long positions in bear ETF (e.g., SOXS, SQQQ, SPXU, TZA, LABD)
- Do not enter bull ETF positions

#### Neutral Regime → Both Allowed but Half Size
**Conditions:**
- Any condition that doesn't meet Bull or Bear criteria

**Trading Rules:**
- Both bull and bear ETFs allowed
- Use **half-size positions** (50% of normal position size)

### 2.3 Technical Indicators for Regime
- **EMA20**: 20-period Exponential Moving Average
- **EMA50**: 50-period Exponential Moving Average
- **Timeframe**: Daily bars (1D)

---

## 3. Entry Setup: Volatility Mean-Reversion Within Trend

### 3.1 Timing
- Triggered **near the last 30 minutes before market close** (3:30 PM ET)
- Decision window: 30 minutes before market close
- Execution: At market close or in extended hours

### 3.2 Indicators on Bull or Bear ETF
- **Bollinger Bands(20, 2)**: 20-period, 2 standard deviations
- **ATR(14)**: 14-period Average True Range
- **RSI fast (2-3)**: Fast Relative Strength Index (2-3 period)
- **RSI slow (14)**: Slow Relative Strength Index (14 period)
- **Volume MA(20)**: 20-day volume moving average

### 3.3 Long Setup Conditions (Bull ETF Example in Bull Regime)

#### 1. Trend Filter
- Must be in **bull regime** (for bull ETF entries, e.g., SOXL, TQQQ, UPRO)
- Must be in **bear regime** (for bear ETF entries, e.g., SOXS, SQQQ, SPXU)

#### 2. Location (Price Near Support)
**Either:**
- Price touches **lower Bollinger Band**, OR
- Price within **0.5-0.8 ATR** of channel support

#### 3. Momentum Exhaustion
- **RSI_fast < 10**
- **RSI_slow < 45**

#### 4. Volume Confirmation
- **Volume ≥ 70%** of 20-day average

### 3.4 Mirror Rules for Bear ETF
- Same logic as bull ETF but for bear regime
- Look for price near **upper Bollinger Band** or upper channel resistance
- RSI conditions: RSI_fast > 90, RSI_slow > 70 (for counter-trend)

---

## 4. Probabilistic Target + Stop Design

### 4.1 ATR-Based Probability Analysis
Use ATR-based hits to determine win/loss probabilities through backtesting.

#### Backtest Process (Updated Monthly)
For each valid setup in last **3-5 years**:
- Track whether **+1 ATR**, **+1.5 ATR**, etc., hit before **-1 ATR**
- Compute probabilities:
  - **p1** = P(+1 ATR hit first)
  - **p2** = P(+1.5 ATR hit first)

### 4.2 Expectancy Formula
If risking **1 ATR** and targeting **1.5 ATR**:

```
EV = p2 × 1.5 - (1 - p2) × 1
```

**Trading Rule:**
- Trade only when **EV > 0**
- Example: p2 = 0.55 → EV = +0.375 R (good trade)

### 4.3 Target and Stop Levels
- **Stop Loss**: -1 ATR from average entry price
- **Target 1 (TP1)**: +1 ATR from average entry price (sell 60% of position)
- **Target 2 (TP2)**: +1.5 ATR from average entry price (sell remaining 40%)

---

## 5. Execution: Laddered Orders

### 5.1 Entry Ladders
**Definition:**
- **P_ref** = Price at decision window (30 min before close)
- **ATR_d** = ATR(14) at decision time

**Place 2 limit buys:**
- **Buy 1 (60%)** → P_ref - 0.3 × ATR_d
- **Buy 2 (40%)** → P_ref - 0.7 × ATR_d

**Execution:**
- Orders are **GTC (Good Till Canceled)** limit orders
- First ladder fills at higher price (closer to P_ref)
- Second ladder fills at lower price (further from P_ref)
- Average entry price = weighted average of both fills

### 5.2 OCO Exit Brackets
**Based on average entry price:**
- **Stop**: -1 ATR (OCO bracket)
- **TP1**: +1 ATR (sell 60% of position)
- **TP2**: +1.5 ATR (sell remaining 40%)

**Order Type:**
- **OCO (One-Cancels-Other)**: Stop and targets are mutually exclusive
- If stop hits, cancel targets
- If TP1 hits, reduce position by 60%, keep stop and TP2 active
- If TP2 hits, close remaining 40%

### 5.3 Forced Exit Rule
- If **neither target nor stop** hit after **5 trading days**, close at market
- Prevents positions from lingering indefinitely

---

## 6. Counter-Trend Bear ETF Flip (Optional, Low Frequency)

### 6.1 Entry Conditions
Enter bear ETF (e.g., SOXS, SQQQ, SPXU) when:
- Bull ETF (e.g., SOXL, TQQQ, UPRO) near **upper Bollinger Band** or upper channel resistance
- **RSI_fast > 90** and **RSI_slow > 70**
- Broad market overheated but still bullish

### 6.2 Counter-Trend Rules
- **Half-size or smaller** positions
- **Stop**: 1 ATR above entry (for bear ETF long)
- **Target**: 1-1.2 ATR
- **Hold**: 1-3 days (shorter than normal)

### 6.3 Frequency
- **Rare**: 1-2 trades per month
- **Avoid overuse** - only when conditions are extreme

---

## 7. Position Sizing (Risk-Based)

### 7.1 Basic Formula
**Variables:**
- **E** = Equity
- **r** = Risk per trade (0.75% of E)
- **D** = Stop distance in % (ATR-based)

**Position Size:**
```
Position Size = (r × E) / D
```

### 7.2 Kelly Fraction (Optional)
Apply **0.25×Kelly** using p2 probability:

```
f_kelly = (p2 × 1.5 - (1 - p2) × 1) / 1.5
f_pos = 0.25 × f_kelly
```

**Final Position Size:**
- Use Kelly-adjusted size if configured
- **Cap final position** to avoid oversized trades
- Respect maximum leverage constraints

### 7.3 Risk Management
- **Maximum per-trade risk**: 0.5-1.0% of account equity
- **Maximum gross leverage**: 1.5x (configurable)
- **Never exceed** account equity limits

---

## 8. Weekly Filter (Higher-Level Safety)

### 8.1 Weekly Trend Filter
**Trade only when:**
- **Bull ETF entries** (e.g., SOXL, TQQQ, UPRO) → Weekly close above **10-week MA**
- **Bear ETF entries** (e.g., SOXS, SQQQ, SPXU) → Weekly close below **10-week MA**

### 8.2 Event-Based Filters
**Reduce size or skip trades if:**
- **FOMC** (Federal Open Market Committee) meetings within 24 hours
- **CPI** (Consumer Price Index) releases within 24 hours
- **Major semiconductor earnings** within 24 hours (NVDA, AMD, INTC, etc.)

### 8.3 Sentiment Filter
**CNN Fear & Greed Index:**
- **>80** → Avoid bull ETF fresh longs (overbought, e.g., SOXL, TQQQ, UPRO)
- **<20** → Avoid bear ETF fresh longs (oversold, e.g., SOXS, SQQQ, SPXU)

---

## 9. Full Playbook Summary

### 9.1 Bull ETF (Bull Regime) - Main Strategy
1. Identify **bull regime** via underlying index (e.g., SOXX, QQQ, SPY) (Price > EMA50, EMA20 > EMA50)
2. Check bull ETF (e.g., SOXL, TQQQ, UPRO) for **lower band touch** or **ATR proximity to support**
3. Confirm **RSI_fast < 10** and **RSI_slow < 45**
4. Confirm **Volume ≥ 70%** of 20-day average
5. Deploy **laddered entries** (60% at P_ref - 0.3×ATR, 40% at P_ref - 0.7×ATR)
6. Use **ATR-based stop and targets** (-1 ATR stop, +1 ATR TP1, +1.5 ATR TP2)
7. Exit remaining after **5 days** if no target hit

### 9.2 Bear ETF (Bear Regime)
- Mirror bull ETF logic during downtrends
- Use **upper Bollinger Band** or upper channel resistance for entries
- RSI conditions: **RSI_fast > 90**, **RSI_slow > 70** (for mean reversion from overbought)

### 9.3 Counter-Trend Reversal Trades
- Rare bull ETF rejection → bear ETF short-term long
- Very small size, fast exits (1-3 days)

---

## 10. Implementation Approaches

### 10.1 Rule-Based Implementation (Current Python)
- **Hardcoded logic** for regime detection, entry conditions, exits
- **Deterministic** - same inputs always produce same outputs
- **Fast execution** - no external API calls
- **Transparent** - easy to debug and understand

### 10.2 AI-Agent Implementation (TypeScript Reference)
- **OpenAI GPT-4** analyzes charts and market context
- **Flexible** - can adapt to market conditions
- **Chart-based** - uses visual chart analysis
- **Numeric features** - EMAs, ATR, RSI provided as ground truth
- **Structured output** - JSON schema for regime, signals, target positions

### 10.3 Hybrid Approach (Recommended)
- **Rule-based regime detection** (fast, deterministic)
- **AI-agent for entry/exit refinement** (flexible, adaptive)
- **Fallback to rules** if AI unavailable

---

## 11. Data Requirements

### 11.1 Market Data
- **Daily bars (1D)** for underlying index, bull ETF, and bear ETF (e.g., SOXX/SOXL/SOXS, QQQ/TQQQ/SQQQ)
- **OHLCV data** (Open, High, Low, Close, Volume)
- **Minimum history**: 60+ days for indicators (EMA50, ATR14, BB20, RSI14)

### 11.2 Indicators to Calculate
- **EMA20, EMA50** (for regime detection on underlying index, e.g., SOXX, QQQ, SPY)
- **Bollinger Bands(20, 2)** (on bull/bear ETFs, e.g., SOXL/SOXS, TQQQ/SQQQ)
- **ATR(14)** (on bull/bear ETFs)
- **RSI(2-3)** and **RSI(14)** (on bull/bear ETFs)
- **Volume MA(20)** (on bull/bear ETFs)
- **Weekly MA(10)** (for weekly filter)

### 11.3 Market Context (Optional but Recommended)
- **CNN Fear & Greed Index** (0-100)
- **FOMC meeting dates**
- **CPI release dates**
- **Semiconductor earnings calendar** (NVDA, AMD, INTC, etc.)

---

## 12. Order Management

### 12.1 Entry Orders
- **Type**: Limit orders (GTC)
- **Ladder 1**: 60% of position at P_ref - 0.3×ATR
- **Ladder 2**: 40% of position at P_ref - 0.7×ATR
- **Timing**: Place 30 minutes before market close

### 12.2 Exit Orders
- **Stop Loss**: Market order triggered at -1 ATR
- **TP1**: Limit order at +1 ATR (sell 60% of position)
- **TP2**: Limit order at +1.5 ATR (sell remaining 40%)
- **Forced Exit**: Market order after 5 trading days

### 12.3 Order Types
- **Limit Orders**: For entries and take profits
- **Market Orders**: For stops and forced exits
- **OCO Brackets**: Stop and targets are mutually exclusive

---

## 13. Performance Metrics

### 13.1 Key Metrics
- **Total Return %**: (Final Equity / Initial Equity - 1) × 100
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average R Multiple**: Average risk/reward ratio per trade

### 13.2 Trade Statistics
- **Number of trades**: Total entries
- **Average holding period**: Days per trade
- **Trade frequency**: Trades per week/month
- **Largest win/loss**: Best and worst trades

---

## 14. Risk Management

### 14.1 Position-Level Risk
- **Maximum risk per trade**: 0.5-1.0% of equity
- **Stop loss**: -1 ATR from entry
- **Forced exit**: 5 trading days maximum

### 14.2 Portfolio-Level Risk
- **Maximum gross leverage**: 1.5x (configurable)
- **Never hold SOXL and SOXS simultaneously** in same direction
- **Neutral regime**: Half-size positions only

### 14.3 Event Risk
- **Skip trades** before major events (FOMC, CPI, earnings)
- **Reduce size** in extreme sentiment (Fear & Greed >80 or <20)

---

## 15. Automation Requirements

### 15.1 Decision Window
- **Time**: 30 minutes before market close (3:30 PM ET)
- **Frequency**: Daily (on trading days only)
- **Inputs**: Current bars, indicators, market context

### 15.2 Execution Window
- **Time**: Market close (4:00 PM ET) or extended hours
- **Orders**: GTC limit orders for entries
- **Monitoring**: Check for fills, update positions

### 15.3 Position Management
- **Daily monitoring**: Check stop/target hits
- **Forced exit**: Close after 5 trading days
- **Revaluation**: Update equity daily

---

## 16. Backtesting Requirements

### 16.1 Data Requirements
- **Start date**: 2018-01-01 (or earliest available)
- **End date**: Current date
- **Resolution**: Daily bars (1D)
- **Symbols**: Underlying index, bull ETF, bear ETF (e.g., SOXX/SOXL/SOXS, QQQ/TQQQ/SQQQ)

### 16.2 Simulation Parameters
- **Initial equity**: $100,000 (configurable)
- **Commission**: $0-1 per trade (configurable)
- **Slippage**: 0-5 bps (basis points, configurable)
- **Fill logic**: At close price (for backtesting)

### 16.3 Validation
- **Compare against Buy & Hold**: Bull ETF benchmark (e.g., SOXL, TQQQ, UPRO)
- **Walk-forward analysis**: Test on out-of-sample data
- **Monte Carlo simulation**: Test robustness

---

## 17. Reference Implementation Notes

### 17.1 TypeScript AI-Agent Approach
- Uses **OpenAI GPT-4** for regime detection and signal generation
- **Chart payloads** (images/HTML) as input to AI
- **Numeric features** (EMAs, ATR, RSI) provided as ground truth
- **Structured JSON output** for regime, signals, target positions

### 17.2 Python Rule-Based Approach (Current)
- **Deterministic logic** for all decisions
- **Fast execution** - no external API dependencies
- **Transparent** - easy to understand and debug
- **Configurable** - all parameters in YAML

### 17.3 Integration Points
- **Data engine**: MarketData.app or yfinance
- **Execution engine**: Simulated (backtest) or live broker API
- **Visualization**: Plotly charts with signals and indicators
- **Caching**: Apache Parquet for historical data

---

## 18. Future Enhancements

### 18.1 AI-Agent Integration
- Add OpenAI integration for flexible regime detection
- Use chart analysis for entry/exit refinement
- Hybrid approach: Rules + AI validation

### 18.2 Advanced Features
- **Weekly filter implementation**: 10-week MA check
- **Event calendar integration**: FOMC, CPI, earnings
- **Sentiment integration**: CNN Fear & Greed Index
- **Counter-trend trades**: Rare SOXS flips

### 18.3 Performance Optimization
- **Parallel backtesting**: Multiple timeframes simultaneously
- **Walk-forward optimization**: Parameter tuning
- **Monte Carlo analysis**: Risk assessment

---

## Appendix A: Configuration Parameters

### A.1 Regime Detection
```yaml
regime_symbol: SOXX  # Underlying index, e.g., SOXX, QQQ, SPY, IWM, XBI
ema_fast: 20
ema_slow: 50
```

### A.2 Trading Instruments
```yaml
bull_etf_symbol: SOXL  # e.g., SOXL, TQQQ, UPRO, TNA, LABU
bear_etf_symbol: SOXS  # e.g., SOXS, SQQQ, SPXU, TZA, LABD
```

### A.3 Indicators
```yaml
bb_length: 20
bb_std: 2.0
atr_length: 14
rsi_fast: 3
rsi_slow: 14
volume_ma_length: 20
```

### A.4 Entry Conditions
```yaml
rsi_fast_threshold: 10.0
rsi_slow_threshold: 45.0
volume_threshold: 0.7
atr_support_range_min: 0.5
atr_support_range_max: 0.8
```

### A.5 Entry Ladders
```yaml
entry_ladder_1_pct: 0.6
entry_ladder_1_atr_offset: 0.3
entry_ladder_2_pct: 0.4
entry_ladder_2_atr_offset: 0.7
```

### A.6 Exits
```yaml
stop_atr_multiple: 1.0
target_1_atr_multiple: 1.0
target_2_atr_multiple: 1.5
max_holding_days: 5
```

### A.7 Position Sizing
```yaml
risk_per_trade_pct: 0.75
kelly_fraction: 0.25
target_probability: 0.55
capital_deployment_pct: 1.0
```

---

## Appendix B: Glossary

- **ATR**: Average True Range - volatility measure
- **BB**: Bollinger Bands - volatility bands around price
- **EMA**: Exponential Moving Average
- **EV**: Expected Value - probability-weighted outcome
- **GTC**: Good Till Canceled - order type
- **OCO**: One-Cancels-Other - bracket order type
- **RSI**: Relative Strength Index - momentum oscillator
- **R Multiple**: Risk/reward ratio (profit/loss in units of risk)

---

## Document History

- **v1.0** (2025-11-27): Initial specification based on PDF and TypeScript reference implementation

