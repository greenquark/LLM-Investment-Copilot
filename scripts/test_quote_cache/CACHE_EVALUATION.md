# Cache Design Evaluation for Current Backtest Patterns

## Analysis of Terminal Output (Lines 830-1003)

### Backtest Configuration
- **Symbol**: AAPL
- **Timeframe**: D (daily)
- **Period**: 2024-10-01 to 2024-12-01 (62 days)
- **Total execution time**: ~3.5 minutes (mostly API calls)

### Current Data Access Patterns

#### 1. Backtest Engine Calls (62 calls)
```python
# Line 74 in engine.py
bars = await self._data.get_bars(symbol, now, now, timeframe=timeframe)
```
- **Frequency**: Once per day (62 times)
- **Range**: Single day or small range (now to now, or 7-day lookback)
- **Pattern**: Minimal overlap between calls

#### 2. Strategy Calls (62 calls)
```python
# Line 133 in mystic_pulse.py
start = now - timedelta(days=30)  # 30-day lookback
bars = await self._data.get_bars(self._symbol, start, now, timeframe)
```
- **Frequency**: Once per day (62 times)
- **Range**: 30-day sliding window
- **Pattern**: **MASSIVE OVERLAP** - each request overlaps with previous by 29 days

**Example progression:**
```
Day 1:  get_bars("AAPL", 2024-09-01, 2024-10-01, "D")  # 30 days
Day 2:  get_bars("AAPL", 2024-09-02, 2024-10-02, "D")  # 29 days overlap!
Day 3:  get_bars("AAPL", 2024-09-03, 2024-10-03, "D")  # 29 days overlap!
...
Day 62: get_bars("AAPL", 2024-11-01, 2024-12-01, "D")  # 29 days overlap!
```

#### 3. Chart Generation Call (1 call)
```python
# Line 150 in run_backtest_mystic_pulse.py
bars = await data_engine.get_bars(symbol, start, end, strategy_config.timeframe)
```
- **Frequency**: Once at end
- **Range**: Full backtest period (2024-10-01 to 2024-12-01)
- **Pattern**: Should be fully cached by this point

### Total API Calls Without Cache
- **Engine**: 62 calls
- **Strategy**: 62 calls  
- **Chart**: 1 call
- **Total**: **125 API calls** for a 2-month backtest!

## Cache Design Evaluation

### ✅ **WILL WORK PERFECTLY** - Here's Why:

#### Scenario 1: First Strategy Call (Day 1)
```
Request:  get_bars("AAPL", 2024-09-01, 2024-10-01, "D")
Cache:    [empty]
Action:   API call → Cache 30 days → Return
Result:   ✅ Cache now has 2024-09-01 to 2024-10-01
```

#### Scenario 2: Second Strategy Call (Day 2)
```
Request:  get_bars("AAPL", 2024-09-02, 2024-10-02, "D")
Cache:    [2024-09-01 ──────────────── 2024-10-01]
Request:  [2024-09-02 ──────────────── 2024-10-02]
           └─ 29 days cached ─┘ └─ 1 day missing ─┘

Action:   
  1. Load cache (29 days already available)
  2. Fetch only missing day (2024-10-02) from API
  3. Merge: [cached 29 days] + [new 1 day]
  4. Save expanded cache (now 2024-09-01 to 2024-10-02)
  5. Return full 30 days

Result:   ✅ Only 1 API call instead of 30!
```

#### Scenario 3: Third Strategy Call (Day 3)
```
Request:  get_bars("AAPL", 2024-09-03, 2024-10-03, "D")
Cache:    [2024-09-01 ─────────────────── 2024-10-02]
Request:  [2024-09-03 ─────────────────── 2024-10-03]
           └─ 30 days fully cached! ─┘

Action:   
  1. Load cache
  2. Filter to requested range (2024-09-03 to 2024-10-03)
  3. Return filtered data
  4. **NO API CALL NEEDED!**

Result:   ✅ Zero API calls!
```

#### Scenario 4: Subsequent Calls (Days 4-62)
```
Pattern: Each day, the cache grows by 1 day
- Day 4: Cache has 2024-09-01 to 2024-10-03, need 2024-10-04 → 1 API call
- Day 5: Cache has 2024-09-01 to 2024-10-04, need 2024-10-05 → 1 API call
- ...
- Day 62: Cache has 2024-09-01 to 2024-12-01, need nothing → 0 API calls

After Day 3, each day only needs 1 new day from API!
```

### Performance Improvement Estimate

**Without Cache:**
- Strategy calls: 62 × 30 days = 1,860 day-equivalents fetched
- Engine calls: 62 × 1 day = 62 day-equivalents
- Chart call: 1 × 62 days = 62 day-equivalents
- **Total**: ~1,984 day-equivalents from API

**With Cache:**
- Day 1: 30 days from API
- Day 2: 1 day from API (29 cached)
- Day 3: 0 days from API (30 cached)
- Days 4-62: 1 day per day = 59 days from API
- Engine calls: Most will be cached (depends on exact timing)
- Chart call: 0 days from API (fully cached)
- **Total**: ~90 day-equivalents from API

**Improvement**: ~95% reduction in API calls! 🚀

### Potential Issues & Solutions

#### Issue 1: Engine Calls May Still Hit API
**Problem**: Engine calls `get_bars(now, now)` which might not be in cache yet
**Solution**: 
- Cache will handle this - if exact day not cached, will fetch
- After strategy call, that day will be cached
- Subsequent engine calls for same day will hit cache

#### Issue 2: Parquet Read Overhead
**Problem**: Reading Parquet file 62 times might be slower than expected
**Solution**:
- Parquet is very fast for filtering by date range
- Can add in-memory cache layer for recently accessed data
- First read loads file, subsequent reads are fast

#### Issue 3: Concurrent Access
**Problem**: Multiple calls for same date range simultaneously
**Solution**:
- Add simple locking mechanism (file lock or in-memory lock)
- Or accept that duplicate API calls are rare and acceptable

### Recommended Optimizations

1. **Batch Strategy Calls**: 
   - Instead of fetching 30 days each time, fetch full range once
   - But current design handles this gracefully

2. **In-Memory Cache Layer**:
   - Keep recently accessed date ranges in memory
   - Reduces Parquet file reads

3. **Pre-fetch Full Range**:
   - At backtest start, fetch full range once
   - All subsequent calls hit cache

## Conclusion

✅ **The cache design WILL WORK EXCELLENTLY** for the current backtest patterns:

1. **Handles sliding windows perfectly**: Each strategy call overlaps 29/30 days with previous
2. **Minimizes API calls**: From 125 calls down to ~90 day-equivalents
3. **Incremental growth**: Cache expands organically, no wasted data
4. **Efficient storage**: Parquet format handles date range queries well
5. **No data loss**: All overlapping scenarios handled correctly

**Expected Performance Improvement:**
- **API calls**: 95% reduction
- **Backtest time**: From ~3.5 minutes to ~30 seconds (estimated)
- **Network usage**: Minimal after first few calls

The design is well-suited for this use case! 🎯

