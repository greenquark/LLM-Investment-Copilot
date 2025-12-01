# Market Data Caching Implementation Plan

## Overview
Implement a file-based caching system for market data to reduce API calls and improve performance. The cache will store historical bar data locally and only fetch from API when data is missing or outdated.

## Design Decisions

### 1. File Format: Apache Parquet
**Rationale:**
- **Efficient**: Columnar format with excellent compression (typically 80-90% size reduction)
- **Fast I/O**: Optimized for read/write operations, especially for time series data
- **Query-friendly**: Can efficiently filter by date ranges without loading entire file
- **Standard**: Widely used in data science and financial applications
- **Type-safe**: Preserves data types (datetime, float) without conversion overhead

**Alternative considered:**
- JSON: Easy but slow and large files
- CSV: Simple but no compression and slow parsing
- SQLite: Good but overkill for simple time series storage
- Pickle: Fast but Python-specific and not human-readable

### 2. Multi-Timeframe Support

**Yes, the cache fully supports multiple timeframes/resolutions:**
- Each `{symbol}_{timeframe}` combination has its own cache file
- Examples:
  - `AAPL_D.parquet` - Daily bars for AAPL
  - `AAPL_15m.parquet` - 15-minute bars for AAPL
  - `AAPL_W.parquet` - Weekly bars for AAPL
  - `TSLA_1m.parquet` - 1-minute bars for TSLA

**Benefits:**
- Independent caching per timeframe (no mixing of resolutions)
- Efficient storage (each timeframe optimized separately)
- Fast queries (only load the specific timeframe needed)
- No data conversion needed (each file stores native resolution)

**Usage:**
When you request `get_bars("AAPL", start, end, timeframe="D")`, it will:
1. Check cache file `AAPL_D.parquet`
2. When you request `get_bars("AAPL", start, end, timeframe="15m")`, it checks `AAPL_15m.parquet`
3. These are completely independent caches

### 3. Cache Structure

#### Directory Layout
```
data_cache/
  ├── bars/
  │   ├── AAPL_D.parquet
  │   ├── AAPL_W.parquet
  │   ├── TSLA_15m.parquet
  │   └── ...
  └── metadata/
      └── cache_index.json  (optional, for quick lookups)
```

#### File Naming Convention
`{SYMBOL}_{TIMEFRAME}.parquet`
- Example: `AAPL_D.parquet`, `AAPL_W.parquet`, `TSLA_15m.parquet`, `SPY_1m.parquet`
- Normalized: Uppercase symbol, timeframe as-is (D, W, or 15m, 1m, 5m, etc.)
- **Multi-timeframe support**: Each symbol+timeframe combination gets its own cache file
  - Same symbol with different timeframes = separate files (e.g., `AAPL_D.parquet` and `AAPL_15m.parquet`)
  - This allows efficient storage and retrieval of different resolutions independently

#### Data Schema (Parquet Columns)
- `timestamp`: datetime64[ns] (indexed for fast range queries)
- `open`: float64
- `high`: float64
- `low`: float64
- `close`: float64
- `volume`: float64
- `symbol`: string (redundant but useful for validation)
- `timeframe`: string (redundant but useful for validation)

### 4. Cache Logic Flow

```
1. Check if cache file exists for {symbol}_{timeframe}
   ├─ NO → Fetch from API → Save to cache → Return data
   └─ YES → Continue to step 2

2. Load cache file and check date range coverage
   ├─ Requested range fully covered → Return cached data
   ├─ Partial coverage → 
   │   ├─ Return cached portion
   │   ├─ Fetch missing data from API
   │   ├─ Merge and deduplicate
   │   └─ Save updated cache
   └─ No coverage (requested range outside cache) →
       ├─ Fetch from API
       ├─ Merge with existing cache (if needed)
       └─ Save updated cache
```

### 4.1 Handling Overlapping Date Ranges

**Scenario: Same symbol+timeframe, different start/end times**

The cache intelligently handles overlapping and non-overlapping date ranges:

#### Example 1: Request Extends Backwards
```
Cache:     [2024-06-01 ──────────────── 2024-12-31]
Request:   [2024-01-01 ──────────────── 2024-12-31]
           └─ Missing ─┘ └─ Cached ───────────────┘

Action:
1. Load cached data (2024-06-01 to 2024-12-31)
2. Fetch missing data from API (2024-01-01 to 2024-05-31)
3. Merge: [API data] + [cached data]
4. Save expanded cache (now covers 2024-01-01 to 2024-12-31)
5. Return full requested range
```

#### Example 2: Request Extends Forwards
```
Cache:     [2024-01-01 ──────────────── 2024-06-30]
Request:   [2024-01-01 ──────────────── 2024-12-31]
           └─ Cached ───────────────┘ └─ Missing ─┘

Action:
1. Load cached data (2024-01-01 to 2024-06-30)
2. Fetch missing data from API (2024-07-01 to 2024-12-31)
3. Merge: [cached data] + [API data]
4. Save expanded cache (now covers 2024-01-01 to 2024-12-31)
5. Return full requested range
```

#### Example 3: Request is Subset of Cache
```
Cache:     [2024-01-01 ──────────────────────────── 2024-12-31]
Request:   [2024-06-01 ──────────────── 2024-09-30]
           └─ Fully covered by cache ─┘

Action:
1. Load cached data
2. Filter to requested range (2024-06-01 to 2024-09-30)
3. Return filtered data (no API call needed)
4. Cache remains unchanged
```

#### Example 4: Request Partially Overlaps
```
Cache:     [2024-01-01 ──────────────── 2024-06-30]
Request:   [2024-04-01 ──────────────── 2024-09-30]
           └─ Overlap ─┘ └─ Missing ─┘

Action:
1. Load cached data (2024-01-01 to 2024-06-30)
2. Filter to overlap (2024-04-01 to 2024-06-30) - use cached
3. Fetch missing data from API (2024-07-01 to 2024-09-30)
4. Merge: [cached overlap] + [API data]
5. Save expanded cache (now covers 2024-01-01 to 2024-09-30)
6. Return full requested range
```

#### Example 5: Request is Completely Outside Cache
```
Cache:     [2024-01-01 ──────────────── 2024-06-30]
Request:   [2025-01-01 ──────────────── 2025-06-30]
           └─ No overlap ─┘

Action:
1. Fetch from API (2025-01-01 to 2025-06-30)
2. Merge with existing cache (append new data)
3. Save expanded cache (now covers 2024-01-01 to 2025-06-30)
4. Return requested range
```

**Key Features:**
- **Deduplication**: If timestamps overlap, cached data takes precedence (assumed more reliable)
- **Efficient filtering**: Uses Parquet's columnar format to only read relevant date ranges
- **Incremental growth**: Cache expands organically as new date ranges are requested
- **No data loss**: All cached data is preserved when merging with new data

**Merge & Deduplication Logic:**
When merging cached and new API data:
1. **Same timestamp exists in both**: Prefer cached data (assumed already validated)
2. **Sort by timestamp**: Ensure chronological order
3. **Remove exact duplicates**: Same symbol + timestamp + timeframe
4. **Preserve all unique bars**: Keep all bars from both sources

Example merge:
```
Cached:  [2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04]
New API: [2024-01-03, 2024-01-04, 2024-01-05, 2024-01-06]
         └─ Overlap ─┘

Result:  [2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05, 2024-01-06]
         └─ Cached ────────────────┘ └─ Cached (preferred) ┘ └─ New ─┘
```

### 5. Implementation Components

#### Component 1: `DataCache` Class
**Location**: `core/data/cache.py`

**Responsibilities:**
- File I/O operations (read/write Parquet)
- Date range checking and filtering
- Cache file path management
- Data merging and deduplication

**Key Methods:**
```python
class DataCache:
    def __init__(self, cache_dir: str = "data_cache/bars")
    
    def get_cache_path(symbol: str, timeframe: str) -> Path
    
    async def load_cached_bars(
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> List[Bar] | None
    
    async def save_bars(
        symbol: str, 
        timeframe: str, 
        bars: List[Bar]
    ) -> None
    
    def _merge_and_deduplicate(
        existing: List[Bar], 
        new: List[Bar]
    ) -> List[Bar]
    """
    Merge cached and new bars, handling overlaps intelligently.
    
    Strategy:
    - If same timestamp exists in both: prefer cached (assumed more reliable)
    - Sort by timestamp
    - Remove duplicates (same symbol+timestamp+timeframe)
    """
    
    def _check_coverage(
        cached_bars: List[Bar], 
        start: datetime, 
        end: datetime
    ) -> CoverageResult
```

**CoverageResult:**
```python
@dataclass
class CoverageResult:
    fully_covered: bool
    partial_covered: bool
    cached_start: datetime | None  # Start of cached data range
    cached_end: datetime | None     # End of cached data range
    missing_ranges: List[Tuple[datetime, datetime]]  # Gaps to fetch from API
    overlapping_bars: List[Bar]  # Bars that match requested range
```

**Coverage Detection Logic:**
```python
def _check_coverage(
    cached_bars: List[Bar], 
    start: datetime, 
    end: datetime
) -> CoverageResult:
    if not cached_bars:
        return CoverageResult(
            fully_covered=False,
            partial_covered=False,
            cached_start=None,
            cached_end=None,
            missing_ranges=[(start, end)],
            overlapping_bars=[]
        )
    
    cached_start = min(b.timestamp for b in cached_bars)
    cached_end = max(b.timestamp for b in cached_bars)
    
    # Filter bars within requested range
    overlapping = [b for b in cached_bars if start <= b.timestamp <= end]
    
    # Check if fully covered
    if overlapping and cached_start <= start and cached_end >= end:
        return CoverageResult(
            fully_covered=True,
            partial_covered=False,
            cached_start=cached_start,
            cached_end=cached_end,
            missing_ranges=[],
            overlapping_bars=overlapping
        )
    
    # Determine missing ranges
    missing_ranges = []
    if cached_start > start:
        # Need data before cache
        missing_ranges.append((start, min(cached_start, end)))
    if cached_end < end:
        # Need data after cache
        missing_ranges.append((max(cached_end, start), end))
    
    return CoverageResult(
        fully_covered=False,
        partial_covered=len(overlapping) > 0,
        cached_start=cached_start,
        cached_end=cached_end,
        missing_ranges=missing_ranges,
        overlapping_bars=overlapping
    )
```

#### Component 2: Modified `MarketDataAppAdapter`
**Location**: `core/data/marketdata_app.py`

**Changes:**
- Add `DataCache` instance in `__init__`
- Modify `get_bars()` to check cache first
- Only call API when cache miss or partial coverage
- Save fetched data to cache

**New Flow:**
```python
async def get_bars(...):
    # 1. Try cache first
    cached = await self._cache.load_cached_bars(symbol, timeframe, start, end)
    if cached and coverage.fully_covered:
        return cached
    
    # 2. Determine what to fetch from API
    if cached:
        # Partial coverage - fetch missing ranges
        missing_ranges = coverage.missing_ranges
    else:
        # No cache - fetch requested range
        missing_ranges = [(start, end)]
    
    # 3. Fetch from API for missing ranges
    api_bars = []
    for range_start, range_end in missing_ranges:
        bars = await self._fetch_from_api(symbol, range_start, range_end, timeframe)
        api_bars.extend(bars)
    
    # 4. Merge cached + API data
    all_bars = self._cache._merge_and_deduplicate(cached or [], api_bars)
    
    # 5. Save to cache
    if api_bars:
        await self._cache.save_bars(symbol, timeframe, all_bars)
    
    # 6. Filter to requested range and return
    return [b for b in all_bars if start <= b.timestamp <= end]
```

### 6. Dependencies

**New Dependencies:**
- `pandas>=2.0.0` - For DataFrame operations and Parquet I/O
- `pyarrow>=14.0.0` - Parquet file format support (fastest implementation)

**Why pandas?**
- Built-in Parquet support via pyarrow
- Efficient DataFrame operations for filtering/merging
- Easy conversion to/from Bar objects

### 7. Data Conversion

**Bar → DataFrame:**
```python
def bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'timestamp': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume,
            'symbol': b.symbol,
            'timeframe': b.timeframe,
        }
        for b in bars
    ]).set_index('timestamp')
```

**DataFrame → Bar:**
```python
def dataframe_to_bars(df: pd.DataFrame, symbol: str, timeframe: str) -> List[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=idx,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe=timeframe,
        )
        for idx, row in df.iterrows()
    ]
```

### 8. Edge Cases & Error Handling

1. **Corrupted cache file**: Delete and re-fetch from API
2. **Partial write failure**: Use atomic writes (write to temp, then rename)
3. **Concurrent access**: File locking or single-writer assumption (acceptable for backtesting)
4. **Disk space**: Log warning but continue (don't fail)
5. **Invalid date ranges**: Validate and handle gracefully
6. **Timezone issues**: Ensure all timestamps are timezone-aware

### 9. Performance Optimizations

1. **Indexed reads**: Use Parquet's columnar format to only read needed date ranges
2. **Batch writes**: Accumulate multiple API responses before writing
3. **Lazy loading**: Only load cache file when needed
4. **Compression**: Use Parquet's built-in compression (snappy or gzip)

### 10. Configuration

**Add to config or environment:**
- `CACHE_DIR`: Default `data_cache/bars`
- `CACHE_ENABLED`: Default `True` (can disable for testing)
- `CACHE_MAX_SIZE_MB`: Optional size limit for cache cleanup

### 11. Testing Strategy

1. **Unit tests**: DataCache class methods
2. **Integration tests**: Full flow with mock API
3. **Performance tests**: Compare cache vs API speed
4. **Edge case tests**: Corrupted files, missing data, etc.

## Example: Multi-Timeframe Usage

```python
# Request daily bars - uses AAPL_D.parquet cache
daily_bars = await data_engine.get_bars("AAPL", start, end, timeframe="D")

# Request 15-minute bars - uses AAPL_15m.parquet cache (separate file)
minute_bars = await data_engine.get_bars("AAPL", start, end, timeframe="15m")

# Request weekly bars - uses AAPL_W.parquet cache (another separate file)
weekly_bars = await data_engine.get_bars("AAPL", start, end, timeframe="W")
```

All three requests use independent cache files, so:
- Caching daily bars doesn't affect 15-minute cache
- Each timeframe is optimized independently
- You can have different date ranges cached for each timeframe

## Implementation Steps

1. **Step 1**: Add dependencies (pandas, pyarrow) to `pyproject.toml`
2. **Step 2**: Create `core/data/cache.py` with `DataCache` class
3. **Step 3**: Implement basic read/write operations
4. **Step 4**: Implement date range checking and merging logic
5. **Step 5**: Integrate cache into `MarketDataAppAdapter`
6. **Step 6**: Add configuration options
7. **Step 7**: Add error handling and logging
8. **Step 8**: Test with real backtest scenarios
9. **Step 9**: Add `.gitignore` entry for cache directory

## File Structure After Implementation

```
core/data/
  ├── __init__.py
  ├── base.py
  ├── cache.py          # NEW: DataCache class
  ├── marketdata_app.py  # MODIFIED: Add cache integration
  ├── moomoo_data.py
  └── wheel_view.py

data_cache/              # NEW: Cache directory (gitignored)
  └── bars/
      ├── AAPL_D.parquet
      └── ...
```

## Benefits

1. **Speed**: 10-100x faster than API calls for cached data
2. **Reliability**: Works offline for cached data
3. **Cost**: Reduces API usage and potential rate limits
4. **Development**: Faster iteration during strategy development
5. **Scalability**: Can cache years of data efficiently

## Future Enhancements (Out of Scope)

1. Cache invalidation strategies (TTL, manual refresh)
2. Cache statistics and monitoring
3. Multi-symbol batch operations
4. Cache compression levels configuration
5. Distributed cache (Redis, etc.) for production

