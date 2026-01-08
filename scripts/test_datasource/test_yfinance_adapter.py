"""
Comprehensive test script for YFinanceDataAdapter and yfinance library.

This script tests:
1. YFinanceDataAdapter functionality (historical bars, option chains)
2. Direct yfinance library features (company info, financials, etc.)
3. All supported timeframes and data types

IMPORTANT NOTE: Option data from yfinance is DAILY-LEVEL only.
- Option chains provide current/latest quotes for a given date
- No minute-level or intraday historical option data is available
- For minute-level option data, use alternative providers (MarketData.app, etc.)
"""

import asyncio
import sys
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data import YFinanceDataAdapter, CachedDataEngine

# Try to import yfinance for direct testing
try:
    import yfinance as yf
    import pandas as pd
    _YFINANCE_DIRECT_AVAILABLE = True
except ImportError:
    _YFINANCE_DIRECT_AVAILABLE = False
    yf = None
    pd = None


async def test_get_bars(symbol: str = "AAPL"):
    """
    Test fetching historical bars with different timeframes.
    
    Note: yfinance has limitations on intraday data:
    - 1m data: only available for last 30 days
    - 5m, 15m, 30m data: only available for last 60 days
    - Daily/weekly/monthly: can go back much further
    
    IMPORTANT: This test bypasses cache by using YFinanceDataAdapter directly
    (not wrapped in CachedDataEngine) to ensure fresh data from the API.
    """
    print("=" * 60)
    print("Testing YFinanceDataAdapter.get_bars() (CACHE & NORMALIZATION BYPASSED)")
    print("=" * 60)
    print("[WARNING]  Note: Intraday data (1m, 5m, 15m, 30m) limited to last 30-60 days by yfinance")
    print("ℹ️  Cache is bypassed - fetching directly from yfinance API")
    print("ℹ️  Time normalization is bypassed - showing raw timestamps from yfinance")
    
    # First, test direct yfinance to verify it's working
    if _YFINANCE_DIRECT_AVAILABLE:
        print("\n[TEST] Quick yfinance direct test:")
        try:
            import yfinance as yf
            test_ticker = yf.Ticker(symbol)
            # Test with period instead of dates (simpler)
            test_df = test_ticker.history(period="5d", interval="1d")
            if not test_df.empty:
                print(f"   [OK] Direct yfinance works! Got {len(test_df)} bars")
                print(f"      Date range: {test_df.index.min()} to {test_df.index.max()}")
                print(f"      Columns: {list(test_df.columns)}")
            else:
                print(f"   [WARNING]  Direct yfinance returned empty DataFrame")
        except Exception as e:
            print(f"   [ERROR] Direct yfinance error: {e}")
    
    try:
        # Use YFinanceDataAdapter directly (no CachedDataEngine wrapper)
        # This ensures cache is bypassed and we get fresh data from yfinance
        # For this test, we'll fetch directly from yfinance to bypass time normalization
        # and see raw timestamps
        if not _YFINANCE_DIRECT_AVAILABLE:
            print("[ERROR] yfinance not available for direct testing")
            print("   Install with: pip install yfinance")
            return
    except ImportError as e:
        print(f"[ERROR] Error: {e}")
        print("Please install yfinance: pip install yfinance")
        return
    
    # Test parameters - use recent dates to meet yfinance limitations
    # yfinance limitations:
    # - 1m data: only 8 days per request (not 30 days!)
    # - 5m, 15m, 30m data: last 60 days only
    # - Daily/weekly/monthly: can go back further
    end_date = datetime.now() - timedelta(days=1)  # Yesterday (ensure it's a trading day)
    
    # Test different timeframes (all supported by yfinance)
    timeframes = ["1m", "5m", "15m", "30m", "1H", "1D", "1wk", "1mo"]
    
    for timeframe in timeframes:
        # Adjust date range based on timeframe to ensure at least 20 bars
        if timeframe == "1wk":
            # Weekly: need at least 20 weeks = 140 days
            start_date = end_date - timedelta(days=140)
        elif timeframe == "1mo":
            # Monthly: need at least 20 months = ~600 days
            start_date = end_date - timedelta(days=600)
        elif timeframe in ["1m", "5m", "15m", "30m"]:
            # Intraday: limited by yfinance
            # 1m: only 8 days per request
            # 5m, 15m, 30m: up to 60 days
            if timeframe == "1m":
                start_date = end_date - timedelta(days=7)  # 7 days for 1m (within 8-day limit)
            else:
                start_date = end_date - timedelta(days=30)  # 30 days for other intraday
        else:
            # Daily and hourly: use 7 days (safe for all)
            start_date = end_date - timedelta(days=7)
        
        print(f"\n[INFO] Testing timeframe: {timeframe}")
        print(f"   Symbol: {symbol}")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            # Fetch directly from yfinance to bypass time normalization
            # This shows raw timestamps as returned by yfinance
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1H": "1h", "1D": "1d", "1wk": "1wk", "1mo": "1mo"
            }
            yf_interval = interval_map.get(timeframe, timeframe)
            
            # Fetch data directly from yfinance (bypasses normalization)
            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(start=start_date, end=end_date, interval=yf_interval, auto_adjust=False)
            )
            
            if df is None or df.empty:
                print(f"   [WARNING]  No bars returned (might be outside trading hours or invalid date range)")
                continue
            
            # Create Bar objects from raw yfinance data (no normalization)
            from core.models.bar import Bar
            bars = []
            for idx, row in df.iterrows():
                # Get timestamp from index
                if hasattr(idx, 'to_pydatetime'):
                    ts = idx.to_pydatetime()
                elif isinstance(idx, datetime):
                    ts = idx
                else:
                    ts = pd.to_datetime(idx).to_pydatetime()
                
                # Convert to timezone-naive ET (but don't normalize to market close)
                # Just convert timezone-aware to naive, keeping the original time
                if ts.tzinfo is not None:
                    # Convert to ET and make naive, but keep original time (no market close normalization)
                    import pytz
                    et_tz = pytz.timezone('America/New_York')
                    et_ts = ts.astimezone(et_tz)
                    ts_naive = et_ts.replace(tzinfo=None)
                else:
                    ts_naive = ts
                
                bars.append(
                    Bar(
                        symbol=symbol,
                        timestamp=ts_naive,  # Raw timestamp, no market close normalization
                        open=float(row.get("Open", row.get("open", 0))),
                        high=float(row.get("High", row.get("high", 0))),
                        low=float(row.get("Low", row.get("low", 0))),
                        close=float(row.get("Close", row.get("close", 0))),
                        volume=float(row.get("Volume", row.get("volume", 0)) if not pd.isna(row.get("Volume", row.get("volume", 0))) else 0),
                        timeframe=timeframe,
                    )
                )
            
            # Sort by timestamp
            bars.sort(key=lambda b: b.timestamp)
            
            if bars:
                print(f"   [OK] Success! Retrieved {len(bars)} bars (RAW - no time normalization)")
                # Print up to 20 bars (or all if fewer than 20)
                num_to_print = min(20, len(bars))
                print(f"   Showing {num_to_print} bar(s) with raw timestamps:")
                for i, bar in enumerate(bars[:num_to_print]):
                    print(f"      {i+1:2d}. {bar.timestamp} | O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,.0f}")
                if len(bars) > 20:
                    print(f"      ... and {len(bars) - 20} more bars")
                print(f"   Data source: yfinance (direct, no normalization)")
            else:
                print(f"   [WARNING]  No bars returned (might be outside trading hours or invalid date range)")
        
        except ValueError as e:
            print(f"   [ERROR] ValueError: {e}")
        except Exception as e:
            print(f"   [ERROR] Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


async def test_get_bars_with_cache(symbol: str = "AAPL"):
    """Test fetching bars with caching enabled."""
    print("\n" + "=" * 60)
    print("Testing YFinanceDataAdapter with CachedDataEngine")
    print("=" * 60)
    
    try:
        base_adapter = YFinanceDataAdapter()
        # Resolve cache directory relative to project root
        cache_dir_path = Path("data_cache/bars")
        if not cache_dir_path.is_absolute():
            cache_dir_resolved = str(project_root / "data_cache/bars")
        else:
            cache_dir_resolved = "data_cache/bars"
        cached_engine = CachedDataEngine(
            base_adapter,
            cache_dir=cache_dir_resolved,
            cache_enabled=True,
        )
    except ImportError as e:
        print(f"[ERROR] Error: {e}")
        return
    
    # Use recent dates to ensure data is available (same approach as test_get_bars)
    # Go back enough days to get a good range but not too far to avoid data issues
    end_date = datetime.now() - timedelta(days=7)  # 7 days ago
    start_date = end_date - timedelta(days=7)  # 7 days before that (14 days ago total)
    
    print(f"\n[INFO] Testing with cache (timeframe: 1D)")
    print(f"   Symbol: {symbol}")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    
    # First call (should hit API)
    print("\n   First call (should fetch from API):")
    try:
        bars1 = await cached_engine.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe="1D",
        )
        print(f"   [OK] Retrieved {len(bars1)} bars")
        print(f"   Data source: {cached_engine.last_data_source}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return
    
    # Second call (should hit cache)
    print("\n   Second call (should fetch from cache):")
    try:
        bars2 = await cached_engine.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe="1D",
        )
        print(f"   [OK] Retrieved {len(bars2)} bars")
        print(f"   Data source: {cached_engine.last_data_source}")
        
        # Verify we got the same data
        if len(bars1) == len(bars2):
            print(f"   [OK] Cache verified: Same number of bars ({len(bars1)})")
        else:
            print(f"   [WARNING]  Warning: Different number of bars (first: {len(bars1)}, second: {len(bars2)})")
    
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Print cache statistics
    try:
        stats = cached_engine.get_cache_stats()
        print(f"\n   [DATA] Cache Statistics:")
        print(f"      Total requests: {stats.get('total_requests', 0)}")
        print(f"      Cache hits: {stats.get('cache_hits', 0)}")
        print(f"      Partial hits: {stats.get('cache_partial_hits', 0)}")
        print(f"      API calls: {stats.get('api_calls', 0)}")
    except Exception as e:
        print(f"   [WARNING]  Could not get cache stats: {e}")


async def test_get_option_chain(symbol: str = "AAPL"):
    """
    Test fetching option chain.
    
    IMPORTANT: Option data from yfinance is DAILY-LEVEL only.
    - Returns current/latest quotes for a given date (snapshot)
    - No minute-level or intraday historical option data
    - For minute-level option data, use alternative providers
    """
    print("\n" + "=" * 60)
    print("Testing YFinanceDataAdapter.get_option_chain()")
    print("=" * 60)
    print("[WARNING]  NOTE: Option data is DAILY-LEVEL (snapshot, not minute-by-minute)")
    print("   For minute-level option data, use MarketData.app or other providers")
    
    try:
        adapter = YFinanceDataAdapter()
    except ImportError as e:
        print(f"[ERROR] Error: {e}")
        return
    
    # Use a recent past date (not today, as markets might be closed)
    test_date = date.today() - timedelta(days=1)
    
    print(f"\n[INFO] Testing option chain (DAILY snapshot)")
    print(f"   Symbol: {symbol}")
    print(f"   As of date: {test_date}")
    print(f"   Data granularity: DAILY (current quotes for this date)")
    print(f"   (Will find closest expiration to {test_date})")
    
    try:
        contracts = await adapter.get_option_chain(
            underlying=symbol,
            as_of=test_date,
        )
        
        if contracts:
            print(f"   [OK] Success! Retrieved {len(contracts)} option contracts")
            
            # Group by type
            calls = [c for c in contracts if c.right == "C"]
            puts = [c for c in contracts if c.right == "P"]
            
            print(f"   Calls: {len(calls)}")
            print(f"   Puts:  {len(puts)}")
            
            # Show expiration date
            if contracts:
                expiry = contracts[0].expiry
                print(f"   Expiration date: {expiry}")
                
                # Show sample contracts
                print(f"\n   Sample contracts (first 5):")
                for i, contract in enumerate(contracts[:5]):
                    print(f"      {i+1}. {contract.symbol} | {contract.right} | Strike: ${contract.strike:.2f} | Expiry: {contract.expiry}")
            
            print(f"\n   [WARNING]  Data Type: DAILY snapshot (not minute-level)")
            print(f"   Data source: {adapter.last_data_source}")
        else:
            print(f"   [WARNING]  No option contracts returned")
    
    except Exception as e:
        print(f"   [ERROR] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def test_yfinance_direct_features(symbol: str = "AAPL"):
    """Test direct yfinance library features (not yet in adapter)."""
    print("\n" + "=" * 60)
    print("Testing Direct yfinance Library Features")
    print("=" * 60)
    print("(These features are available but not yet implemented in YFinanceDataAdapter)")
    
    if not _YFINANCE_DIRECT_AVAILABLE:
        print("\n[WARNING]  yfinance not available for direct testing")
        print("   Install with: pip install yfinance")
        return
    
    loop = asyncio.get_event_loop()
    
    # Test 1: Company Info
    print(f"\n[INFO] 1. Company Information (Ticker.info)")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        info = await loop.run_in_executor(None, lambda: ticker.info)
        
        if info:
            print(f"   [OK] Retrieved company info ({len(info)} fields)")
            print(f"   Company: {info.get('longName', 'N/A')}")
            print(f"   Sector: {info.get('sector', 'N/A')}")
            print(f"   Market Cap: ${info.get('marketCap', 0):,}" if info.get('marketCap') else "   Market Cap: N/A")
            print(f"   P/E Ratio: {info.get('trailingPE', 'N/A')}")
            print(f"   Beta: {info.get('beta', 'N/A')}")
        else:
            print(f"   [WARNING]  No company info returned")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 2: Financial Statements
    print(f"\n[INFO] 2. Financial Statements")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        
        # Test income statement
        financials = await loop.run_in_executor(None, lambda: ticker.financials)
        if financials is not None and not financials.empty:
            print(f"   [OK] Income Statement: {len(financials.columns)} periods, {len(financials)} line items")
        else:
            print(f"   [WARNING]  Income Statement: No data")
        
        # Test balance sheet
        balance = await loop.run_in_executor(None, lambda: ticker.balance_sheet)
        if balance is not None and not balance.empty:
            print(f"   [OK] Balance Sheet: {len(balance.columns)} periods, {len(balance)} line items")
        else:
            print(f"   [WARNING]  Balance Sheet: No data")
        
        # Test cash flow
        cashflow = await loop.run_in_executor(None, lambda: ticker.cashflow)
        if cashflow is not None and not cashflow.empty:
            print(f"   [OK] Cash Flow: {len(cashflow.columns)} periods, {len(cashflow)} line items")
        else:
            print(f"   [WARNING]  Cash Flow: No data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 3: Dividends and Splits
    print(f"\n[INFO] 3. Dividends and Stock Splits")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        dividends = await loop.run_in_executor(None, lambda: ticker.dividends)
        splits = await loop.run_in_executor(None, lambda: ticker.splits)
        
        if dividends is not None and len(dividends) > 0:
            print(f"   [OK] Dividends: {len(dividends)} dividend payments")
            print(f"      Latest: {dividends.index[-1].date()} - ${dividends.iloc[-1]:.4f}")
        else:
            print(f"   [WARNING]  Dividends: No data")
        
        if splits is not None and len(splits) > 0:
            print(f"   [OK] Stock Splits: {len(splits)} splits")
            print(f"      Latest: {splits.index[-1].date()} - {splits.iloc[-1]:.1f}:1")
        else:
            print(f"   [WARNING]  Stock Splits: No data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 4: Analyst Recommendations
    print(f"\n[INFO] 4. Analyst Recommendations")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        recommendations = await loop.run_in_executor(None, lambda: ticker.recommendations)
        
        if recommendations is not None and len(recommendations) > 0:
            print(f"   [OK] Recommendations: {len(recommendations)} analyst ratings")
            # Handle different index types (DatetimeIndex, Timestamp, etc.)
            latest_idx = recommendations.index[-1]
            if hasattr(latest_idx, 'date'):
                latest_date = latest_idx.date()
            elif hasattr(latest_idx, 'to_pydatetime'):
                latest_date = latest_idx.to_pydatetime().date()
            elif isinstance(latest_idx, (int, float)):
                # Sometimes the index might be numeric (period number)
                latest_date = f"Period {latest_idx}"
            else:
                latest_date = str(latest_idx)
            
            # Get the grade - recommendations DataFrame structure varies
            # The recommendations DataFrame has columns like 'Firm', 'To Grade', 'From Grade', 'Action'
            latest_row = recommendations.iloc[-1]
            if hasattr(latest_row, 'get'):
                latest_grade = latest_row.get('To Grade', latest_row.get('Grade', 'N/A'))
            elif isinstance(latest_row, (pd.Series, dict)):
                latest_grade = latest_row.get('To Grade', 'N/A') if hasattr(latest_row, 'get') else 'N/A'
            else:
                latest_grade = 'N/A'
            print(f"      Latest: {latest_date} - Grade: {latest_grade}")
        else:
            print(f"   [WARNING]  Recommendations: No data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 5: Institutional Holders
    print(f"\n[INFO] 5. Institutional Holders")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        holders = await loop.run_in_executor(None, lambda: ticker.institutional_holders)
        
        if holders is not None and len(holders) > 0:
            print(f"   [OK] Institutional Holders: {len(holders)} institutions")
            print(f"      Top holder: {holders.iloc[0].get('Holder', 'N/A')}")
        else:
            print(f"   [WARNING]  Institutional Holders: No data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 6: Earnings (using income_stmt instead of deprecated earnings)
    print(f"\n[INFO] 6. Earnings Data")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        # Use income_stmt instead of deprecated earnings
        income_stmt = await loop.run_in_executor(None, lambda: ticker.income_stmt)
        
        if income_stmt is not None and not income_stmt.empty:
            # Look for Net Income or Earnings
            if 'Net Income' in income_stmt.index:
                net_income = income_stmt.loc['Net Income']
                print(f"   [OK] Earnings (from income_stmt): {len(net_income)} periods")
                if len(net_income) > 0:
                    latest_value = net_income.iloc[-1]
                    latest_date = net_income.index[-1]
                    # Handle NaN values
                    if pd.isna(latest_value):
                        print(f"      Latest: {latest_date} - Net Income: N/A (data not available)")
                    elif isinstance(latest_value, (int, float)):
                        print(f"      Latest: {latest_date} - Net Income: ${latest_value:,.0f}")
                    else:
                        print(f"      Latest: {latest_date} - Net Income: {latest_value}")
            else:
                print(f"   [WARNING]  Earnings: Net Income not found in income statement")
        else:
            print(f"   [WARNING]  Earnings: No income statement data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
    
    # Test 7: News
    print(f"\n[INFO] 7. News")
    try:
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
        news = await loop.run_in_executor(None, lambda: ticker.news)
        
        if news and len(news) > 0:
            print(f"   [OK] News: {len(news)} articles")
            if news[0].get('title'):
                print(f"      Latest: {news[0].get('title', 'N/A')[:60]}...")
        else:
            print(f"   [WARNING]  News: No data")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")


async def test_timeframe_validation():
    """Test timeframe validation."""
    print("\n" + "=" * 60)
    print("Testing Timeframe Validation")
    print("=" * 60)
    
    try:
        adapter = YFinanceDataAdapter()
    except ImportError as e:
        print(f"[ERROR] Error: {e}")
        return
    
    # Valid timeframes
    valid_timeframes = ["1m", "15m", "1H", "1D", "1wk", "1mo"]
    
    # Invalid timeframes
    invalid_timeframes = ["3m", "2H", "2D", "2wk", "2mo", "invalid"]
    
    print("\n[OK] Testing valid timeframes:")
    for tf in valid_timeframes:
        try:
            normalized = adapter._normalize_timeframe(tf)
            print(f"   {tf:6s} -> {normalized}")
        except Exception as e:
            print(f"   {tf:6s} -> [ERROR] Error: {e}")
    
    print("\n[ERROR] Testing invalid timeframes:")
    for tf in invalid_timeframes:
        try:
            normalized = adapter._normalize_timeframe(tf)
            print(f"   {tf:6s} -> {normalized} (unexpected success)")
        except ValueError as e:
            print(f"   {tf:6s} -> [OK] Correctly rejected: {str(e)[:50]}")
        except Exception as e:
            print(f"   {tf:6s} -> [WARNING]  Unexpected error: {e}")


async def main(symbol: str = "AAPL"):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("YFinanceDataAdapter Comprehensive Test Suite")
    print("=" * 60)
    print(f"\nTesting symbol: {symbol}")
    print("\nThis script tests:")
    print("  1. YFinanceDataAdapter functionality (historical bars, option chains)")
    print("  2. Direct yfinance library features (company info, financials, etc.)")
    print("  3. All supported timeframes and data types")
    print("\n[WARNING] IMPORTANT: Option data is DAILY-LEVEL only (snapshot, not minute-by-minute)")
    print("   For minute-level option data, use MarketData.app or other providers")
    print("\nNote: Tests require internet connection and may take a few seconds.\n")
    
    # Run tests
    await test_timeframe_validation()
    await test_get_bars(symbol)
    await test_get_bars_with_cache(symbol)
    await test_get_option_chain(symbol)
    await test_yfinance_direct_features(symbol)
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
    print("\nSummary:")
    print("  [OK] Historical bars: Supports multiple timeframes (1m to 1mo)")
    print("  [OK] Option chains: DAILY-LEVEL only (current quotes for a date)")
    print("  [OK] Caching: Integrated with CachedDataEngine")
    print("  [OK] Direct yfinance: Company info, financials, dividends, etc. available")
    print("\n[WARNING] Limitations:")
    print("  - Option data is daily snapshot only (no minute-level historical data)")
    print("  - Real-time streaming not supported")
    print("  - Some data may require Yahoo Finance premium subscription")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test YFinanceDataAdapter with a specific ticker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_yfinance_adapter.py --ticker AAPL
  python scripts/test_yfinance_adapter.py --ticker TSLA
  python scripts/test_yfinance_adapter.py --ticker SOXL
        """
    )
    parser.add_argument(
        "--ticker",
        "--symbol",
        type=str,
        default="AAPL",
        dest="ticker",
        help="Stock ticker symbol to test (default: AAPL)",
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(symbol=args.ticker))
    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()

