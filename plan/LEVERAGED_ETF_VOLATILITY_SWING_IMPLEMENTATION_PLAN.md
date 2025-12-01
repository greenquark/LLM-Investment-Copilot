# Leveraged ETF Volatility Swing Strategy - Implementation Plan

**Last Updated**: 2025-01-27  
**Status**: Phase 1 Core Complete, Debugging & Improvements In Progress  
**Next Phase**: Phase 1.5 (Bug Fixes & Visualization) → Phase 2 (Market Context Integration)

---

## Overview

This document outlines the implementation plan for the Leveraged ETF Volatility Swing Trading Strategy. The strategy is designed to work with any bull/bear leveraged ETF pair (e.g., SOXL/SOXS, TQQQ/SQQQ, UPRO/SPXU, TNA/TZA, LABU/LABD).

---

## Current State Assessment

### ✅ Completed (Phase 0 & Phase 1 - Core Implementation)

**What's Already Implemented:**
1. ✅ Rule-based strategy (`core/strategy/leveraged_etf_vol_swing.py`)
   - Regime detection (Bull/Bear/Neutral) using underlying index EMA20/EMA50
   - Entry conditions (Bollinger Bands, ATR, RSI, Volume)
   - Laddered entry logic (60%/40% at different ATR offsets)
   - ATR-based exits (Stop, TP1, TP2)
   - Forced exit after 5 days
   - Position sizing with risk-based calculation
   - Generic naming (works with any leveraged ETF pair)
   - ✅ **Recent**: Enhanced diagnostic logging for entry condition debugging
   - ✅ **Recent**: Data fetching fallback (90-day lookback to avoid API limits)
   - ✅ **Recent**: Improved insufficient data handling with single-log-per-symbol

2. ✅ Backtest script (`scripts/run_backtest_leveraged_etf.py`)
   - Full backtest execution
   - Performance metrics calculation
   - Buy & Hold benchmark comparison
   - Cache statistics display
   - Interactive Plotly charts

3. ✅ Configuration file (`config/strategy.leveraged_etf_vol_swing.yaml`)
   - All strategy parameters configurable
   - Defaults to SOXL/SOXS but supports any pair

4. ✅ Documentation
   - Specification document (`docs/LEVERAGED_ETF_VOLATILITY_SWING_STRATEGY_SPEC.md`)
   - Implementation plan (this document)

5. ✅ **Recent Improvements (2025-01-27)**
   - Date consistency fixes across codebase (pandas Timestamp vs Python date normalization)
   - Cache engine improvements for 100% cache hits on reruns
   - Diagnostic logging for entry condition analysis
   - Data fetching robustness (handles API limits gracefully)

**What's Missing:**
1. ❌ Weekly filter (10-week MA check)
2. ❌ Event calendar integration (FOMC, CPI, earnings)
3. ❌ Sentiment filter (CNN Fear & Greed Index)
4. ❌ Counter-trend bear ETF flip trades
5. ❌ AI agent integration (optional enhancement)
6. ❌ Chart payload generation for AI analysis
7. ❌ Probabilistic target validation (monthly backtest for p1/p2)

**Known Issues:**
1. ⚠️ Indicator history not populated (`_indicator_history` is empty)
   - `IndicatorData` model is designed for Mystic Pulse strategy (positive_count, negative_count, etc.)
   - Need to create `LeveragedETFIndicatorData` model or extend `IndicatorData` for leveraged ETF indicators
   - Should populate with BB, ATR, RSI, Volume MA values when entry setup is detected
2. ⚠️ Forced exit uses calendar days instead of trading days
   - Currently: `(now.date() - self._entry_date.date()).days >= max_holding_days`
   - Should use trading calendar to count actual trading days
3. ⚠️ Ladder entry tracking incomplete
   - `_ladder_1_filled` and `_ladder_2_filled` flags exist but not fully utilized
   - Average entry price calculation uses reference price, not actual fill prices

---

## Phase 1.5: Bug Fixes & Visualization (Priority: High - Current Focus)

### 1.5.1 Indicator History Population
**Status**: Not Started  
**Estimated Time**: 2-3 hours  
**Priority**: High (needed for visualization and debugging)

**Tasks:**
- [ ] Create `LeveragedETFIndicatorData` dataclass in `core/visualization/models.py`:
  - `timestamp: datetime`
  - `price: float`
  - `bb_upper: float`
  - `bb_middle: float`
  - `bb_lower: float`
  - `atr: float`
  - `rsi_fast: float`
  - `rsi_slow: float`
  - `volume: float`
  - `volume_ma: float`
  - `regime: str` (bull/bear/neutral)
  - `entry_setup_detected: bool`
- [ ] Populate `_indicator_history` in `_check_entry_setups()`:
  - Add indicator data for every bar (or every 10th bar to reduce memory)
  - Mark `entry_setup_detected=True` when all conditions met
- [ ] Update `get_indicator_history()` to return correct type
- [ ] Update backtest script to visualize indicator history on charts

**Files to Modify:**
- `core/visualization/models.py`: Add `LeveragedETFIndicatorData`
- `core/strategy/leveraged_etf_vol_swing.py`: Populate indicator history
- `scripts/run_backtest_leveraged_etf.py`: Visualize indicators on charts

**Testing:**
- Unit test for indicator data population
- Verify indicator history appears in charts
- Test memory usage with large backtests

---

### 1.5.2 Trading Days Fix for Forced Exit
**Status**: Not Started  
**Estimated Time**: 1-2 hours  
**Priority**: Medium (affects position management accuracy)

**Tasks:**
- [ ] Use trading calendar to count trading days instead of calendar days
- [ ] Update `_manage_position()` to use `get_trading_days_between()` or similar
- [ ] Ensure forced exit triggers after exactly 5 trading days

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Fix forced exit logic
- `core/utils/trading_days.py`: Add helper function if needed

**Testing:**
- Unit test for trading days calculation
- Integration test verifying forced exit after 5 trading days (not calendar days)

---

## Phase 1: Core Enhancements (Priority: High)

### 1.1 Weekly Filter Implementation
**Status**: Not Started  
**Estimated Time**: 2-3 hours

**Tasks:**
- [ ] Add weekly bar fetching capability to data engine
- [ ] Calculate 10-week MA on weekly bars
- [ ] Add filter check before entry in `_check_entry_setups()`:
  - Bull ETF entries: Weekly close > 10-week MA
  - Bear ETF entries: Weekly close < 10-week MA
- [ ] Skip trades if weekly filter fails
- [ ] Add logging for weekly filter decisions

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Add weekly filter check
- `core/data/base.py`: Ensure weekly timeframe support (verify)
- `config/strategy.leveraged_etf_vol_swing.yaml`: Add `weekly_filter_enabled: bool` parameter

**Testing:**
- Unit test for weekly MA calculation
- Integration test with weekly filter enabled/disabled
- Backtest validation showing weekly filter impact

---

### 1.2 Counter-Trend Bear ETF Flip Trades
**Status**: Not Started  
**Estimated Time**: 3-4 hours

**Tasks:**
- [ ] Add counter-trend entry logic:
  - Bull ETF near upper BB or upper channel resistance
  - RSI_fast > 90, RSI_slow > 70
  - Market overheated but still bullish
- [ ] Use half-size positions for counter-trend trades
- [ ] Shorter holding period (1-3 days)
- [ ] Different stop/target (1 ATR above entry, 1-1.2 ATR target)
- [ ] Add `_check_counter_trend_setup()` method

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Add counter-trend logic
- `LeveragedETFVolSwingConfig`: Add counter-trend parameters:
  - `counter_trend_enabled: bool = False`
  - `counter_trend_size_multiplier: float = 0.5`
  - `counter_trend_max_days: int = 3`
  - `counter_trend_stop_atr_multiple: float = 1.0`
  - `counter_trend_target_atr_multiple: float = 1.1`

**Testing:**
- Unit test for counter-trend setup detection
- Backtest validation showing counter-trend trade frequency
- Verify half-size positions are used

---

### 1.3 Enhanced Position Management
**Status**: Partially Complete  
**Estimated Time**: 2-3 hours  
**Note**: Trading days fix moved to Phase 1.5.2

**Tasks:**
- [ ] Track ladder fills separately (60%/40%)
  - Store actual fill prices for ladder 1 and ladder 2
  - Track `_ladder_1_entry_price` and `_ladder_2_entry_price`
- [ ] Calculate average entry price from actual fills (not reference price)
  - Weighted average: `(ladder_1_price * ladder_1_shares + ladder_2_price * ladder_2_shares) / total_shares`
- [ ] Handle partial exits correctly (TP1 sells 60%, TP2 sells remaining 40%)
  - Ensure TP1 sells from ladder 1 shares first
  - TP2 sells remaining shares
- [ ] Add position state tracking for ladder fills
  - Track which ladder entries have been filled
  - Handle case where ladder 2 never fills (price doesn't drop enough)

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: 
  - Enhance `_execute_laddered_entry()` to track actual fills
  - Update `_manage_position()` to use actual average entry price
  - Add state variables for ladder fill prices

**Testing:**
- Unit test for ladder fill tracking
- Unit test for average entry price calculation
- Integration test for partial exits
- Test scenario where ladder 2 never fills

---

## Phase 2: Market Context Integration (Priority: Medium)

### 2.1 Event Calendar Integration
**Status**: Not Started  
**Estimated Time**: 4-6 hours

**Tasks:**
- [ ] Create `core/data/event_calendar.py`:
  - FOMC meeting dates (quarterly, from Federal Reserve calendar)
  - CPI release dates (monthly, from BLS calendar)
  - Major semiconductor earnings (NVDA, AMD, INTC, etc.)
  - Support for other sector earnings (configurable)
- [ ] Add event check before entry:
  - Skip trades if event within 24 hours
  - Reduce position size if event within 48 hours
- [ ] Create `config/events.yaml` for event calendar configuration

**New Files:**
- `core/data/event_calendar.py`: Event calendar data source
- `config/events.yaml`: Event calendar configuration

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Add event check in `_check_entry_setups()`
- `LeveragedETFVolSwingConfig`: Add event filter parameters:
  - `event_filter_enabled: bool = True`
  - `event_skip_hours: int = 24`
  - `event_reduce_size_hours: int = 48`
  - `event_reduce_size_multiplier: float = 0.5`

**Data Sources:**
- FOMC: Federal Reserve website or API
- CPI: BLS (Bureau of Labor Statistics) calendar
- Earnings: Earnings calendar API or static file

**Testing:**
- Unit test for event detection
- Integration test with events enabled/disabled
- Backtest validation showing event filter impact

---

### 2.2 Sentiment Filter (CNN Fear & Greed Index)
**Status**: Not Started  
**Estimated Time**: 3-4 hours

**Tasks:**
- [ ] Create `core/data/sentiment.py`:
  - Fetch CNN Fear & Greed Index (0-100)
  - Cache daily values
  - Handle API failures gracefully
- [ ] Add sentiment check:
  - Fear & Greed > 80: Avoid bull ETF fresh longs
  - Fear & Greed < 20: Avoid bear ETF fresh longs
- [ ] Create `core/data/fear_greed.py`: CNN Fear & Greed API integration

**New Files:**
- `core/data/sentiment.py`: Sentiment data source wrapper
- `core/data/fear_greed.py`: CNN Fear & Greed API integration (if available)

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Add sentiment check in `_check_entry_setups()`
- `LeveragedETFVolSwingConfig`: Add sentiment filter parameters:
  - `sentiment_filter_enabled: bool = True`
  - `sentiment_avoid_bull_threshold: int = 80`
  - `sentiment_avoid_bear_threshold: int = 20`

**Data Sources:**
- CNN Fear & Greed Index: https://www.cnn.com/markets/fear-and-greed (may need scraping or API)

**Testing:**
- Unit test for sentiment filter logic
- Integration test with sentiment enabled/disabled
- Mock API responses for testing

---

## Phase 3: AI Agent Integration (Priority: Low - Optional)

### 3.1 AI Agent Module
**Status**: Not Started  
**Estimated Time**: 8-12 hours

**Tasks:**
- [ ] Create `core/strategy/ai_agent.py`:
  - OpenAI GPT-4 integration
  - Chart payload generation (Plotly to base64 PNG)
  - Structured JSON output parsing
  - Response validation
- [ ] Create `core/strategy/regime_agent.py`: Regime detection agent (optional)

**New Files:**
- `core/strategy/ai_agent.py`: AI agent wrapper
- `core/strategy/regime_agent.py`: Regime detection agent (optional)

**Dependencies:**
- `openai>=1.0.0`: OpenAI Python SDK
- `plotly>=5.0.0`: Chart generation (already installed)
- `Pillow>=10.0.0`: Image processing for base64 encoding

**Testing:**
- Unit test for AI agent response parsing
- Integration test with mock OpenAI responses
- Validate structured output format

---

### 3.2 Hybrid Strategy Mode
**Status**: Not Started  
**Estimated Time**: 4-6 hours

**Tasks:**
- [ ] Add `use_ai_agent: bool` config option
- [ ] If enabled:
  - Use AI for regime detection and signal refinement
  - Fall back to rule-based if AI unavailable
- [ ] If disabled:
  - Use rule-based logic (current implementation)

**Files to Modify:**
- `core/strategy/leveraged_etf_vol_swing.py`: Add AI agent integration
- `LeveragedETFVolSwingConfig`: Add `use_ai_agent: bool = False` parameter
- `config/strategy.leveraged_etf_vol_swing.yaml`: Add AI config section

**Testing:**
- Integration test with AI enabled/disabled
- Test fallback to rule-based when AI fails
- Compare AI vs rule-based performance

---

### 3.3 Chart Payload Generation
**Status**: Not Started  
**Estimated Time**: 3-4 hours

**Tasks:**
- [ ] Create `core/visualization/chart_payload.py`:
  - Generate Plotly charts for underlying index, bull ETF, bear ETF
  - Convert to base64 PNG or HTML
  - Include indicators (BB, ATR, RSI) on charts

**New Files:**
- `core/visualization/chart_payload.py`: Chart payload generator

**Testing:**
- Unit test for chart generation
- Unit test for base64 encoding
- Validate chart payload format

---

## Phase 4: Probabilistic Target Validation (Priority: Medium)

### 4.1 Monthly Backtest for Probability Calculation
**Status**: Not Started  
**Estimated Time**: 6-8 hours

**Tasks:**
- [ ] Create `scripts/calculate_atr_probabilities.py`:
  - Backtest all valid setups in last 3-5 years
  - Track whether +1 ATR, +1.5 ATR hit before -1 ATR
  - Calculate p1, p2 probabilities
  - Update `target_probability` in config

**New Files:**
- `scripts/calculate_atr_probabilities.py`: Probability calculator
- `core/strategy/probability_calculator.py`: Probability analysis module

**Output:**
- Monthly report with updated probabilities
- Config file update with new `target_probability` value
- CSV/JSON export of probability data

**Testing:**
- Unit test for probability calculation
- Validate against known historical data
- Test config file update logic

---

## Phase 5: Enhanced Backtesting (Priority: Medium)

### 5.1 Walk-Forward Analysis
**Status**: Not Started  
**Estimated Time**: 6-8 hours

**Tasks:**
- [ ] Create `scripts/walk_forward_optimization.py`:
  - Test strategy on rolling windows
  - Optimize parameters (ATR multiples, RSI thresholds)
  - Validate on out-of-sample data

**New Files:**
- `scripts/walk_forward_optimization.py`: Walk-forward optimizer
- `core/backtest/optimization.py`: Parameter optimization utilities

**Testing:**
- Unit test for walk-forward logic
- Validate parameter optimization
- Test out-of-sample validation

---

### 5.2 Monte Carlo Simulation
**Status**: Not Started  
**Estimated Time**: 4-6 hours

**Tasks:**
- [ ] Create `scripts/monte_carlo_backtest.py`:
  - Run multiple backtests with randomized parameters
  - Assess robustness and risk
  - Generate confidence intervals

**New Files:**
- `scripts/monte_carlo_backtest.py`: Monte Carlo simulator

**Testing:**
- Unit test for Monte Carlo logic
- Validate statistical distributions
- Test confidence interval calculation

---

### 5.3 Enhanced Metrics
**Status**: Not Started  
**Estimated Time**: 3-4 hours

**Tasks:**
- [ ] Add trade-level statistics:
  - Win rate, average R multiple
  - Largest win/loss
  - Average holding period
  - Trade frequency (trades per week/month)

**Files to Modify:**
- `core/backtest/performance.py`: Add trade-level metrics
- `scripts/run_backtest_leveraged_etf.py`: Display enhanced metrics

**Testing:**
- Unit test for trade-level metric calculation
- Validate against manual calculations

---

## Phase 6: Live Trading Integration (Priority: Low - Future)

### 6.1 Broker API Integration
**Status**: Not Started  
**Estimated Time**: 12-16 hours

**Tasks:**
- [ ] Create `core/execution/live_broker.py`:
  - Broker API wrapper (Interactive Brokers, Alpaca, etc.)
  - Order submission and monitoring
  - Position tracking

**New Files:**
- `core/execution/live_broker.py`: Live broker integration

**Testing:**
- Unit test for broker API wrapper
- Integration test with paper trading account
- Validate order execution

---

### 6.2 Scheduler for Live Trading
**Status**: Not Started  
**Estimated Time**: 4-6 hours

**Tasks:**
- [ ] Create `core/live/scheduler.py`:
  - Daily scheduler (30 min before close)
  - Decision window execution
  - Order placement and monitoring

**New Files:**
- `core/live/scheduler.py`: Live trading scheduler
- `scripts/run_live_leveraged_etf.py`: Live trading script

**Testing:**
- Unit test for scheduler logic
- Integration test with mock market data
- Validate decision window timing

---

## Implementation Priority Summary

### Immediate (Current Sprint):
1. ⚠️ **Indicator history population** (Phase 1.5.1) - Needed for visualization
2. ⚠️ **Trading days fix for forced exit** (Phase 1.5.2) - Accuracy improvement
3. ⚠️ **Enhanced position management** (Phase 1.3) - Ladder tracking improvements

### Short-term (Week 1-2):
4. Weekly filter implementation (Phase 1.1)
5. Counter-trend bear ETF flip trades (Phase 1.2)

### Short-term (Week 3-4):
4. Event calendar integration
5. Sentiment filter (Fear & Greed)
6. Probabilistic target validation

### Medium-term (Month 2-3):
7. AI agent integration (optional)
8. Walk-forward optimization
9. Monte Carlo simulation

### Long-term (Month 4+):
10. Live trading integration
11. Advanced risk management
12. Multi-timeframe support

---

## Technical Considerations

### Dependencies to Add:
- `openai>=1.0.0` (for AI agent - optional)
- `Pillow>=10.0.0` (for chart image encoding - optional)
- Event calendar data source (API or static file)
- CNN Fear & Greed Index API or scraping library

### Configuration Enhancements:
- Add `weekly_filter_enabled: bool`
- Add `counter_trend_enabled: bool`
- Add `event_filter_enabled: bool`
- Add `sentiment_filter_enabled: bool`
- Add `use_ai_agent: bool` (optional)

### Testing Requirements:
- Unit tests for weekly filter
- Unit tests for counter-trend logic
- Integration tests for event calendar
- Backtest validation against reference implementation

---

## Notes for Future Revisions

1. **Performance Optimization**: Consider parallel backtesting for multiple ETF pairs
2. **Parameter Tuning**: Create automated parameter optimization scripts
3. **Risk Management**: Add portfolio-level risk controls
4. **Multi-Strategy**: Support running multiple ETF pairs simultaneously
5. **Real-time Monitoring**: Add dashboard for live trading monitoring
6. **Alert System**: Notify on trade signals, stops, targets
7. **Backtest Comparison**: Compare performance across different ETF pairs

---

## Revision History

- **2025-01-27**: Updated plan with recent improvements and current issues
  - Added Phase 1.5 (Bug Fixes & Visualization)
  - Documented recent improvements: diagnostic logging, data fetching fallback, date consistency fixes
  - Identified known issues: indicator history not populated, forced exit uses calendar days
  - Updated priorities to focus on visualization and bug fixes first

- **2025-11-27**: Initial implementation plan created
  - Phase 0 & Phase 1 (Core Implementation) completed
  - Strategy renamed from SOXL/SOXS to generic Leveraged ETF
  - Plan document created in `plan/` directory

---

## Next Steps

1. ✅ **Complete Phase 1.5.1**: Implement indicator history population
   - Create `LeveragedETFIndicatorData` model
   - Populate `_indicator_history` in `_check_entry_setups()`
   - Update visualization to show indicators

2. ✅ **Complete Phase 1.5.2**: Fix forced exit to use trading days
   - Use trading calendar for accurate day counting
   - Test with various entry dates

3. **Complete Phase 1.3**: Enhanced position management
   - Track actual ladder fill prices
   - Calculate weighted average entry price
   - Improve partial exit handling

4. **Start Phase 1.1**: Weekly filter implementation
   - Add weekly bar fetching
   - Implement 10-week MA check
   - Add filter to entry conditions

5. **Start Phase 1.2**: Counter-trend bear ETF flip trades
   - Add counter-trend setup detection
   - Implement half-size positions
   - Add shorter holding period logic

6. Move to Phase 2 (Market Context) after Phase 1 is complete

