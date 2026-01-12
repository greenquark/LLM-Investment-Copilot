# Plan: Option Quotes and Option Chains Support

## Design Principles

This implementation follows a **modular, agentic design** to support:
1. **MCP Server Integration** - Expose option data operations as MCP tools for external use
2. **Conversational Interfaces** - Agents can explain their actions and reasoning
3. **Composable Agents** - Option data agents can be used independently or composed with other agents
4. **Modular Components** - Each component is independently usable and testable

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server (TypeScript)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Option Chain │  │ Option Quote│  │ Option Bars  │  │
│  │    Tool      │  │    Tool     │  │    Tool      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │   Python Option Data Agents          │
          │  ┌──────────────┐  ┌──────────────┐ │
          │  │OptionChain   │  │OptionQuote   │ │
          │  │   Agent      │  │   Agent      │ │
          │  └──────┬───────┘  └──────┬───────┘ │
          │         │                  │         │
          │  ┌──────▼──────────────────▼───────┐ │
          │  │   OptionDataService (Core)      │ │
          │  └──────┬───────────────────────────┘ │
          └─────────┼────────────────────────────┘
                    │
          ┌─────────▼─────────┐
          │   DataEngine      │
          │  (MarketData.app) │
          └───────────────────┘
```

## Current State Analysis

### What Exists
1. **Models:**
   - `OptionContract` (basic: symbol, underlying, expiry, strike, right)
   - `OptionDetails` in `Order` model (underlying, expiry, strike, right)

2. **Data Engine:**
   - `MarketDataAppAdapter.get_option_chain()` - Fetches option chains for a date
   - `CachedDataEngine.get_option_chain()` - Pass-through (no caching)
   - `DataEngine` base class defines `get_option_chain()` as abstract

3. **Usage:**
   - `WheelStrategy` uses option chains for CSP/CC strategies
   - `WheelDataView` fetches option chains for context

### What's Missing
1. **Option Quotes:**
   - Real-time/delayed prices (bid, ask, last, mark)
   - Volume and open interest
   - Greeks (delta, gamma, theta, vega, rho)
   - Implied volatility (IV)
   - Historical option prices (OHLC bars)

2. **Enhanced Models:**
   - Option quote model with pricing data
   - Option contract with full details (Greeks, IV, etc.)
   - Historical option bar data

3. **Caching:**
   - Option chain caching (with expiration date as key)
   - Option quote caching (time-sensitive)

4. **API Support:**
   - MarketData.app option quotes endpoint
   - Historical option bars endpoint
   - Real-time option streaming (if supported)

---

## Implementation Plan

### Phase 0: Agent Architecture Design

#### 0.1 Create Option Data Service Layer
**File:** `core/agents/option_data_service.py`

**Purpose:** Centralized service layer that option data agents use. Provides:
- Unified interface for option data operations
- Error handling and retry logic
- Rate limiting coordination
- Logging and observability

**Design:**
```python
class OptionDataService:
    """Service layer for option data operations.
    
    This service provides a unified interface for option data operations,
    making it easy to create agents that work with option data.
    
    Note: Works with any DataEngine (including CachedDataEngine).
    Caching is transparent - service doesn't need to know about it.
    """
    
    def __init__(self, data_engine: DataEngine):
        """
        Initialize with a DataEngine.
        
        For caching, wrap the base engine:
            base_engine = MarketDataAppAdapter(api_token="...")
            cached_engine = CachedDataEngine(base_engine, cache_dir="data_cache")
            service = OptionDataService(cached_engine)
        """
        self._data_engine = data_engine
        self._logger = logging.getLogger(__name__)
    
    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
        include_quotes: bool = False,
    ) -> List[OptionContract]:
        """
        Get option chain with optional quotes.
        
        If data_engine is CachedDataEngine, caching is automatic and transparent.
        """
        ...
    
    async def get_option_quote(
        self,
        contract: OptionContract,
        as_of: Optional[datetime] = None,
    ) -> Optional[OptionQuote]:
        """
        Get quote for a specific option contract.
        
        If data_engine is CachedDataEngine, historical quotes are cached automatically.
        """
        ...
    
    async def get_option_bars(
        self,
        contract: OptionContract,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[OptionBar]:
        """
        Get historical bars for an option contract.
        
        If data_engine is CachedDataEngine, bars are cached automatically
        using the same infrastructure as stock bars.
        """
        ...
```

**Rationale:** Service layer enables:
- Easy composition of agents
- Consistent error handling
- Centralized logging for conversational interfaces
- Future extension to multiple data sources
- **Transparent caching**: Works with `CachedDataEngine` without knowing about it

#### 0.2 Create Option Data Agents
**File:** `core/agents/option_chain_agent.py`

**Purpose:** Agent that can fetch and analyze option chains. Designed for:
- MCP server integration
- Conversational interfaces (can explain what it's doing)
- Composition with other agents

**Design:**
```python
class OptionChainAgent:
    """Agent for fetching and analyzing option chains.
    
    This agent can:
    - Fetch option chains for any underlying
    - Filter chains by expiration, strike, moneyness
    - Explain its actions for conversational interfaces
    - Be composed with other agents (e.g., strategy agents)
    """
    
    def __init__(self, option_service: OptionDataService):
        self._service = option_service
        self._logger = logging.getLogger(__name__)
    
    async def fetch_chain(
        self,
        underlying: str,
        as_of: date,
        explain: bool = True,
    ) -> Tuple[List[OptionContract], str]:
        """
        Fetch option chain.
        
        Returns:
            Tuple of (contracts, explanation) if explain=True
        """
        if explain:
            explanation = f"Fetching option chain for {underlying} as of {as_of}..."
            self._logger.info(explanation)
        
        contracts = await self._service.get_option_chain(underlying, as_of)
        
        if explain:
            explanation += f"\nFound {len(contracts)} contracts across {len(set(c.expiry for c in contracts))} expiration dates."
        
        return contracts, explanation if explain else ""
    
    async def filter_chain(
        self,
        contracts: List[OptionContract],
        expiration_range: Optional[Tuple[date, date]] = None,
        strike_range: Optional[Tuple[float, float]] = None,
        moneyness: Optional[str] = None,  # "ITM", "OTM", "ATM"
        explain: bool = True,
    ) -> Tuple[List[OptionContract], str]:
        """Filter option chain with explanation."""
        ...
```

**File:** `core/agents/option_quote_agent.py`

**Similar design for option quotes.**

**Rationale:** Agents provide:
- Self-documenting behavior (explain actions)
- Composable units that can work together
- Easy MCP integration (agents become tools)

#### 0.3 MCP Server Tools
**File:** `mcp/server.ts`

**New Tools:**
```typescript
server.tool("get_option_chain", {
  description: "Fetch option chain for an underlying symbol on a specific date. Returns list of option contracts with details.",
  inputSchema: z.object({
    underlying: z.string().describe("Underlying symbol (e.g., 'AAPL')"),
    asOf: z.string().describe("Date in ISO format (YYYY-MM-DD)"),
    includeQuotes: z.boolean().optional().describe("Whether to include current quotes"),
    filterExpiration: z.object({
      min: z.string().optional(),
      max: z.string().optional(),
    }).optional(),
    filterStrike: z.object({
      min: z.number().optional(),
      max: z.number().optional(),
    }).optional(),
  }),
  async execute({ input }) {
    // Call Python agent via subprocess or HTTP
    // Return structured data
  },
});

server.tool("get_option_quote", {
  description: "Get current or historical quote for a specific option contract. Returns bid, ask, Greeks, IV, etc.",
  inputSchema: z.object({
    optionSymbol: z.string().describe("Option symbol (OCC format)"),
    asOf: z.string().optional().describe("Timestamp for historical quote (ISO format)"),
  }),
  async execute({ input }) {
    // Call Python agent
  },
});

server.tool("get_option_bars", {
  description: "Get historical OHLC bars for an option contract. Similar to stock bars but for options.",
  inputSchema: z.object({
    optionSymbol: z.string(),
    start: z.string().describe("Start timestamp (ISO format)"),
    end: z.string().describe("End timestamp (ISO format)"),
    timeframe: z.string().describe("Bar timeframe (e.g., '1D', '1H', '15m')"),
  }),
  async execute({ input }) {
    // Call Python agent
  },
});

server.tool("analyze_option_chain", {
  description: "Analyze an option chain and provide insights. Can filter, calculate metrics, and explain findings.",
  inputSchema: z.object({
    underlying: z.string(),
    asOf: z.string(),
    analysis: z.object({
      findBestStrike: z.boolean().optional(),
      calculateGreeks: z.boolean().optional(),
      filterByMoneyness: z.enum(["ITM", "OTM", "ATM"]).optional(),
    }).optional(),
  }),
  async execute({ input }) {
    // Use OptionChainAgent with analysis capabilities
    // Return structured analysis with explanations
  },
});
```

**Rationale:** MCP tools enable:
- External agents to use option data
- Conversational interfaces to interact with option data
- Composition with other trading tools

### Phase 1: Enhanced Option Models

#### 1.1 Extend `OptionContract` Model
**File:** `core/models/option.py`

**Changes:**
- Add optional fields for quote data:
  - `bid: Optional[float]`
  - `ask: Optional[float]`
  - `last: Optional[float]`
  - `mark: Optional[float]`
  - `volume: Optional[int]`
  - `open_interest: Optional[int]`
  - `implied_volatility: Optional[float]`
  - `delta: Optional[float]`
  - `gamma: Optional[float]`
  - `theta: Optional[float]`
  - `vega: Optional[float]`
  - `rho: Optional[float]`
  - `quote_timestamp: Optional[datetime]`

**Rationale:** Single model that can represent both chain contracts and quoted contracts.

#### 1.2 Create `OptionQuote` Model
**File:** `core/models/option.py`

**New Model:**
```python
@dataclass
class OptionQuote:
    """Real-time or delayed option quote with pricing and Greeks."""
    contract: OptionContract
    bid: float
    ask: float
    last: float
    mark: float  # Mid price or theoretical value
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    quote_timestamp: datetime
    underlying_price: float  # Price of underlying at quote time
```

**Rationale:** Separate model for quotes allows for cleaner separation of concerns.

#### 1.3 Create `OptionBar` Model
**File:** `core/models/option.py`

**New Model:**
```python
@dataclass
class OptionBar:
    """Historical OHLC bar for an option contract."""
    contract: OptionContract
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int
    timeframe: str
```

**Rationale:** Enables historical option price analysis similar to stock bars.

---

### Phase 2: Data Engine Extensions

**Note:** This phase implements the core data fetching. Agents (Phase 0) will use these methods.

#### 2.1 Add Option Quote Methods to `DataEngine` Base Class
**File:** `core/data/base.py`

**New Methods:**
```python
@abstractmethod
async def get_option_quote(
    self,
    contract: OptionContract,
    as_of: Optional[datetime] = None,
) -> Optional[OptionQuote]:
    """
    Get current or historical quote for a specific option contract.
    
    Args:
        contract: The option contract to quote
        as_of: Optional timestamp for historical quote (None for current)
    
    Returns:
        OptionQuote if available, None otherwise
    """
    ...

@abstractmethod
async def get_option_quotes(
    self,
    contracts: List[OptionContract],
    as_of: Optional[datetime] = None,
) -> List[OptionQuote]:
    """
    Get quotes for multiple option contracts in batch.
    
    Args:
        contracts: List of option contracts to quote
        as_of: Optional timestamp for historical quotes (None for current)
    
    Returns:
        List of OptionQuote objects (may be shorter than input if some unavailable)
    """
    ...

@abstractmethod
async def get_option_bars(
    self,
    contract: OptionContract,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> List[OptionBar]:
    """
    Get historical OHLC bars for an option contract.
    
    Args:
        contract: The option contract
        start: Start timestamp
        end: End timestamp
        timeframe: Bar timeframe (e.g., "1D", "1H", "15m")
    
    Returns:
        List of OptionBar objects
    """
    ...
```

**Rationale:** Standard interface for all data engines.

#### 2.2 Implement in `MarketDataAppAdapter`
**File:** `core/data/marketdata_app.py`

**Implementation Tasks:**

1. **`get_option_quote()`:**
   - Endpoint: `GET /v1/options/quote/{option_symbol}/`
   - Parameters: `as_of` (optional timestamp)
   - Parse response into `OptionQuote` model
   - Handle missing data gracefully

2. **`get_option_quotes()`:**
   - Batch endpoint if available: `POST /v1/options/quotes/`
   - Or loop through `get_option_quote()` with rate limiting
   - Return list of quotes

3. **`get_option_bars()`:**
   - Endpoint: `GET /v1/options/candles/{timeframe}/{option_symbol}/`
   - Similar structure to `get_bars()` for stocks
   - Parse into `OptionBar` objects

**Error Handling:**
- Handle 404 for expired/invalid contracts
- Retry logic for timeouts (reuse existing pattern)
- Graceful degradation if API doesn't support certain features

#### 2.3 Enhance `get_option_chain()` to Include Quotes
**File:** `core/data/marketdata_app.py`

**Enhancement:**
- Add optional parameter: `include_quotes: bool = False`
- If `True`, fetch quotes for all contracts in chain
- Populate `OptionContract` fields with quote data
- Or return separate list of `OptionQuote` objects

**Trade-off:** 
- Option A: Populate `OptionContract` fields (simpler, but mixes concerns)
- Option B: Return separate `List[OptionQuote]` (cleaner separation)

**Recommendation:** Option B - return separate quotes list for flexibility.

---

### Phase 3: Extend Existing Caching Infrastructure

**Note:** The `CachedDataEngine` already provides transparent caching for any `DataEngine`. We just need to extend `DataCache` to support option data types, and `CachedDataEngine` will automatically use it.

#### 3.1 Extend `DataCache` for Option Chains
**File:** `core/data/cache.py`

**New Methods:**
```python
def get_cache_path_option_chain(
    self,
    underlying: str,
    as_of: date,
) -> Path:
    """Get cache file path for option chain."""
    # Returns: data_cache/option_chains/{underlying}_{as_of.isoformat()}.parquet
    ...

async def load_cached_option_chain(
    self,
    underlying: str,
    as_of: date,
) -> Optional[List[OptionContract]]:
    """Load option chain from cache."""
    # Similar pattern to load_cached_bars()
    ...

async def save_option_chain(
    self,
    underlying: str,
    as_of: date,
    contracts: List[OptionContract],
) -> None:
    """Save option chain to cache."""
    # Similar pattern to save_bars()
    # Convert OptionContract objects to DataFrame, save as Parquet
    ...
```

**Caching Strategy:**
- Cache key: `{underlying}_{as_of.isoformat()}.parquet`
- Cache directory: `data_cache/option_chains/`
- TTL: Option chains are date-specific, so cache indefinitely (user can clear manually)
- Format: Parquet (reuse existing infrastructure)

#### 3.2 Extend `DataCache` for Option Quotes (Optional)
**File:** `core/data/cache.py`

**New Methods:**
```python
def get_cache_path_option_quote(
    self,
    contract: OptionContract,
    as_of: Optional[datetime],
) -> Path:
    """Get cache file path for option quote."""
    ...

async def load_cached_option_quote(
    self,
    contract: OptionContract,
    as_of: Optional[datetime],
) -> Optional[OptionQuote]:
    """Load option quote from cache."""
    ...

async def save_option_quote(
    self,
    contract: OptionContract,
    quote: OptionQuote,
) -> None:
    """Save option quote to cache."""
    ...
```

**Caching Strategy:**
- Cache key: `{option_symbol}_{as_of or 'current'}.parquet`
- Cache directory: `data_cache/option_quotes/`
- TTL: 
  - Current quotes: 5 minutes (very time-sensitive)
  - Historical quotes: Indefinitely
- Format: Parquet

**Note:** Option quotes are highly time-sensitive. Consider whether caching is beneficial or if it adds complexity without much benefit. May skip this initially.

#### 3.3 Extend `DataCache` for Option Bars
**File:** `core/data/cache.py`

**Reuse existing bar caching infrastructure:**
- Option bars can use the same `get_cache_path()`, `save_bars()`, and `load_cached_bars()` methods
- Use `symbol=option_symbol` (e.g., "AAPL240119C00150000")
- May need to extend `Bar` model or create adapter to handle `OptionBar` → `Bar` conversion
- Or extend `save_bars()` to accept `OptionBar` objects directly

**Alternative Approach:**
- Create `OptionBar` → `Bar` adapter/converter
- Store option bars using existing bar cache infrastructure
- Add metadata field to distinguish option bars from stock bars

#### 3.4 Extend `CachedDataEngine` for Option Data
**File:** `core/data/cached_engine.py`

**Update existing `get_option_chain()` method:**
```python
async def get_option_chain(
    self,
    underlying: str,
    as_of: date,
) -> list[OptionContract]:
    """
    Get option chain with transparent caching.
    
    Uses the same pattern as get_bars():
    1. Check cache first
    2. Fetch missing data from base engine
    3. Merge and save to cache
    """
    # Track requests
    self._total_requests += 1
    
    # Check cache if enabled
    if self._cache_enabled and self._cache:
        cached_chain = await self._cache.load_cached_option_chain(underlying, as_of)
        if cached_chain:
            self._cache_hits += 1
            self._last_data_source = "Cache"
            return cached_chain
    
    # Fetch from base engine
    self._base_engine_calls += 1
    self._last_data_source = self._get_data_source_name()
    contracts = await self._base_engine.get_option_chain(underlying, as_of)
    
    # Save to cache
    if self._cache_enabled and self._cache and contracts:
        await self._cache.save_option_chain(underlying, as_of, contracts)
    
    return contracts
```

**Add new methods for option quotes and bars:**
```python
async def get_option_quote(
    self,
    contract: OptionContract,
    as_of: Optional[datetime] = None,
) -> Optional[OptionQuote]:
    """
    Get option quote with optional caching.
    
    Note: Current quotes are time-sensitive, so caching may be disabled by default.
    """
    # Similar pattern to get_option_chain()
    # Check cache for historical quotes, skip for current quotes
    ...

async def get_option_bars(
    self,
    contract: OptionContract,
    start: datetime,
    end: datetime,
    timeframe: str,
) -> List[OptionBar]:
    """
    Get option bars with transparent caching.
    
    Uses the same pattern as get_bars() for stocks.
    """
    # Reuse existing bar caching logic
    # May need to convert OptionBar <-> Bar or extend cache methods
    ...
```

**Key Point:** `CachedDataEngine` already provides the transparent caching wrapper pattern. We just need to:
1. Extend `DataCache` to support option data types
2. Update `CachedDataEngine` methods to use the cache
3. The caching is transparent - agents and services don't need to know about it

---

### Phase 4: API Integration Details

#### 4.1 MarketData.app API Research
**Tasks:**
1. Review MarketData.app documentation for:
   - Option quote endpoints
   - Option historical data endpoints
   - Batch quote endpoints
   - Rate limits
   - Data format (JSON columnar vs row-based)

2. Test API endpoints:
   - Verify response structure
   - Check error handling
   - Measure response times
   - Test with various option symbols

#### 4.2 Implementation Details

**Option Quote Endpoint:**
```
GET /v1/options/quote/{option_symbol}/
GET /v1/options/quote/{option_symbol}/?as_of={timestamp}
```

**Response Format (expected):**
```json
{
  "s": "ok",
  "optionSymbol": "AAPL240119C00150000",
  "bid": 2.50,
  "ask": 2.55,
  "last": 2.52,
  "mark": 2.525,
  "volume": 1500,
  "openInterest": 5000,
  "impliedVolatility": 0.25,
  "delta": 0.65,
  "gamma": 0.02,
  "theta": -0.05,
  "vega": 0.15,
  "rho": 0.01,
  "underlyingPrice": 150.00,
  "timestamp": 1704067200
}
```

**Option Bars Endpoint:**
```
GET /v1/options/candles/{timeframe}/{option_symbol}/?from={start}&to={end}
```

**Response Format (expected):**
Similar to stock candles (columnar format):
```json
{
  "s": "ok",
  "t": [1704067200, 1704153600, ...],
  "o": [2.50, 2.55, ...],
  "h": [2.60, 2.65, ...],
  "l": [2.45, 2.50, ...],
  "c": [2.55, 2.60, ...],
  "v": [1000, 1200, ...],
  "oi": [5000, 5100, ...]  // Open interest
}
```

---

### Phase 5: Testing and Validation

#### 5.1 Unit Tests
**Files:** `tests/test_option_data.py`

**Test Cases:**
1. `test_get_option_chain()` - Verify chain retrieval
2. `test_get_option_quote()` - Verify single quote retrieval
3. `test_get_option_quotes()` - Verify batch quote retrieval
4. `test_get_option_bars()` - Verify historical bars
5. `test_option_chain_caching()` - Verify caching works
6. `test_option_quote_caching()` - Verify quote caching (if implemented)
7. `test_option_bars_caching()` - Verify bars caching
8. `test_error_handling()` - Test 404, timeouts, invalid contracts

#### 5.2 Integration Tests
**Files:** `tests/integration/test_marketdata_options.py`

**Test Cases:**
1. End-to-end option chain retrieval
2. End-to-end quote retrieval
3. End-to-end historical bars retrieval
4. Cache hit/miss scenarios
5. Rate limiting behavior

#### 5.3 Manual Testing
**Test Script:** `scripts/test_option_data.py`

**Features:**
- Interactive script to test option data retrieval
- Display option chains with quotes
- Display individual option quotes
- Display historical option bars
- Test caching behavior

---

### Phase 6: Documentation

#### 6.1 API Documentation
**File:** `docs/OPTION_DATA_API.md`

**Contents:**
- Overview of option data support
- API method signatures
- Usage examples
- Error handling
- Rate limits
- Caching behavior

#### 6.2 Usage Examples
**File:** `docs/OPTION_DATA_EXAMPLES.md`

**Examples:**
1. Fetch option chain
2. Fetch option quotes
3. Fetch historical option bars
4. Use with caching
5. Integration with strategies

---

## Implementation Order

### Priority 0 (Agent Architecture - Foundation)
1. ✅ Phase 0.1: Create `OptionDataService` layer
2. ✅ Phase 0.2: Create `OptionChainAgent` and `OptionQuoteAgent`
3. ✅ Phase 0.3: Add MCP server tools for option data

### Priority 1 (Core Functionality)
4. ✅ Phase 1.1: Extend `OptionContract` model
5. ✅ Phase 1.2: Create `OptionQuote` model
6. ✅ Phase 1.3: Create `OptionBar` model
7. ✅ Phase 2.1: Add methods to `DataEngine` base class
8. ✅ Phase 2.2: Implement in `MarketDataAppAdapter`
9. ✅ Phase 4.1: Research MarketData.app API

### Priority 2 (Extend Existing Caching)
10. ✅ Phase 3.1: Extend `DataCache` for option chains
11. ✅ Phase 3.3: Extend `DataCache` for option bars (reuse bar infrastructure)
12. ✅ Phase 3.4: Extend `CachedDataEngine` methods to use cache
13. ⚠️ Phase 3.2: Option quote caching (optional - may skip due to time-sensitivity)

### Priority 3 (Enhancements)
13. ✅ Phase 2.3: Enhance `get_option_chain()` with quotes
14. ✅ Phase 3.2: Option quote caching (if beneficial)
15. ✅ Phase 5: Testing
16. ✅ Phase 6: Documentation

---

## Considerations

### 1. Modularity & Reusability
- **Service Layer**: `OptionDataService` provides unified interface, making it easy to swap data sources
- **Agent Independence**: Agents can be used standalone or composed together
- **Clear Interfaces**: Each component has well-defined inputs/outputs for easy testing and composition

### 2. Agentic Design
- **Self-Explaining**: Agents can explain their actions (`explain=True` parameter)
- **Composable**: Agents can call other agents (e.g., strategy agent uses option chain agent)
- **Observable**: All actions are logged for conversational interfaces

### 3. MCP Integration
- **Tool Descriptions**: Clear, detailed descriptions for LLM understanding
- **Structured Output**: Return structured data (JSON) for programmatic use
- **Error Messages**: Human-readable error messages for conversational interfaces

### 4. Rate Limiting
- Option quotes may have stricter rate limits than stock data
- Implement rate limiting/throttling in `OptionDataService`
- Consider batch endpoints to reduce API calls
- Agents should explain rate limit issues to users

### 5. Data Freshness
- Option quotes are highly time-sensitive
- Consider whether caching is beneficial or adds complexity
- May want to skip caching for current quotes, only cache historical
- Agents should indicate data freshness in explanations

### 6. Error Handling
- Expired contracts return 404
- Invalid option symbols return errors
- Handle gracefully in strategies
- **Agent-friendly**: Errors should be explainable to conversational interfaces

### 7. Cost
- Option data may have different pricing tiers
- Monitor API usage to avoid unexpected costs
- Agents should warn about expensive operations

### 8. Performance
- Batch quote requests when possible
- **Leverage existing `CachedDataEngine`**: Option chains and bars automatically benefit from transparent caching
- Cache option chains (date-specific, less time-sensitive) - handled by `CachedDataEngine`
- Cache historical option bars (similar to stock bars) - can reuse existing bar cache infrastructure
- Agents should report performance metrics when requested
- **Transparent caching**: Agents don't need to know about caching - `CachedDataEngine` handles it automatically

---

## Future Enhancements

### Agent Enhancements

1. **Option Strategy Agent:**
   - Agent that can analyze option strategies (spreads, straddles, etc.)
   - Calculate P&L, risk/reward
   - Explain strategy mechanics conversationally
   - Composable with option data agents

2. **Option Screening Agent:**
   - Agent that screens options based on criteria
   - Can explain screening logic
   - Returns ranked results with explanations

3. **Real-time Option Monitoring Agent:**
   - WebSocket support for live option quotes
   - Can monitor specific contracts and alert on changes
   - Explains what it's monitoring and why

### Technical Enhancements

4. **Option Greeks Calculations:**
   - Calculate Greeks if API doesn't provide
   - Use Black-Scholes or other models
   - Agent can explain calculation methodology

5. **Option Chain Filtering:**
   - Filter by expiration date range
   - Filter by strike price range
   - Filter by moneyness (ITM/OTM/ATM)
   - Agent explains filtering logic

6. **Option Strategy Analysis:**
   - Calculate P&L for spreads
   - Analyze risk/reward for strategies
   - Visualize option strategies
   - Agent explains analysis results

7. **Multi-Provider Support:**
   - Support other data providers (Polygon, Alpha Vantage, etc.)
   - Fallback mechanisms if primary provider fails
   - Agent can explain which provider is being used and why

8. **Conversational Query Interface:**
   - Natural language queries: "Show me ATM calls for AAPL expiring next week"
   - Agent interprets and executes queries
   - Returns structured data with explanations

9. **Agent Composition Framework:**
   - Framework for composing multiple agents
   - Example: Strategy agent + Option data agent + Risk analysis agent
   - Agents can share context and collaborate

---

## Success Criteria

1. ✅ Can fetch option chains for any underlying and date
2. ✅ Can fetch option quotes (current and historical)
3. ✅ Can fetch historical option bars
4. ✅ Caching works for option chains and bars
5. ✅ Error handling is robust
6. ✅ Performance is acceptable (rate limits respected)
7. ✅ Documentation is complete
8. ✅ Tests pass

---

## Estimated Effort

- **Phase 0 (Agent Architecture):** 6-8 hours
  - Service layer: 2-3 hours
  - Agents: 3-4 hours
  - MCP tools: 1-2 hours
- **Phase 1 (Models):** 2-4 hours
- **Phase 2 (Data Engine):** 8-12 hours
- **Phase 3 (Extend Caching):** 3-5 hours (less work since infrastructure exists)
- **Phase 4 (API Research):** 2-4 hours
- **Phase 5 (Testing):** 4-6 hours
- **Phase 6 (Documentation):** 2-3 hours

**Total:** 28-43 hours

**Note:** The agent architecture (Phase 0) is critical for the modular, agentic design. It adds some upfront effort but enables:
- Easy MCP integration
- Conversational interfaces
- Agent composition
- Future extensibility

---

## Notes

### Design Philosophy
- **Agents First**: Design agents that can explain their actions and be composed
- **Service Layer**: Centralize business logic in services, not agents
- **MCP Integration**: Every major capability should be exposed as an MCP tool
- **Conversational**: Agents should be able to explain what they're doing and why

### Implementation Strategy
- Start with MarketData.app API since it's already integrated
- Build agent architecture first (Phase 0) - enables everything else
- Consider option quote caching carefully - may not be worth the complexity
- Focus on option chains and historical bars first (most useful for backtesting)
- Real-time quotes can be added later if needed for live trading

### Agent Communication
- Agents should use structured logging for observability
- Error messages should be human-readable
- Agents should provide context in their explanations
- Consider adding a "verbose" mode for detailed explanations

### Testing Strategy
- Test agents independently
- Test agent composition
- Test MCP tool integration
- Test conversational interfaces (can agent explain its actions?)

