"""
Test script for Yahoo Finance data source.

This script tests:
1. YFinance data source configuration
2. Fetching historical bars using the configured data source
3. Cache functionality with yfinance
4. Different timeframes and date ranges

Usage:
    python scripts/test_datasource/test_yfinance.py
    python scripts/test_datasource/test_yfinance.py --symbol AAPL --timeframe 1D
"""

import asyncio
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def test_yfinance_data_source(
    symbol: str = "AAPL",
    timeframe: str = "1D",
    days_back: int = 30,
):
    """
    Test yfinance data source with the configured settings.
    
    Args:
        symbol: Stock symbol to test
        timeframe: Timeframe to test (e.g., "1D", "1H", "15m")
        days_back: Number of days to fetch historical data
    """
    print("=" * 80)
    print("YAHOO FINANCE DATA SOURCE TEST")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Days back: {days_back}")
    print()
    
    # Load configuration
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    
    if not env_file.exists():
        print(f"❌ Config file not found: {env_file}")
        return
    
    print(f"[INFO] Loading configuration from: {env_file}")
    env = load_config_with_secrets(env_file)
    
    # Check data source configuration
    data_sources = env.get("data_sources", {})
    data_config = env.get("data", {})
    
    print("\n[INFO] Data Source Configuration:")
    print(f"  MarketData.app enabled: {data_sources.get('marketdata_app', {}).get('enabled', False)}")
    print(f"  YFinance enabled: {data_sources.get('yfinance', {}).get('enabled', False)}")
    print(f"  Historical source: {data_config.get('historical_source', 'N/A')}")
    print(f"  Cache enabled: {data_config.get('cache_enabled', False)}")
    print(f"  Cache directory: {data_config.get('cache_dir', 'N/A')}")
    
    if data_config.get('historical_source') != 'yfinance':
        print(f"\n[WARNING] historical_source is set to '{data_config.get('historical_source')}', not 'yfinance'")
        print("   Update config/env.backtest.yaml to use yfinance")
    
    if not data_sources.get('yfinance', {}).get('enabled', False):
        print(f"\n[WARNING] yfinance is not enabled in configuration")
        print("   Update config/env.backtest.yaml to enable yfinance")
    
    # Resolve cache directory relative to project root
    cache_dir_from_config = data_config.get('cache_dir', 'data_cache/bars')
    cache_dir_path = Path(cache_dir_from_config)
    if not cache_dir_path.is_absolute():
        # Make it relative to project root - use resolve() to normalize the path
        cache_dir_resolved = str((project_root / cache_dir_from_config).resolve())
    else:
        # If already absolute, just normalize it
        cache_dir_resolved = str(Path(cache_dir_from_config).resolve())
    
    print(f"\n[INFO] Cache directory (resolved): {cache_dir_resolved}")
    print(f"[INFO] Project root: {project_root}")
    
    # Create data engine from config
    print("\n[INFO] Creating data engine from configuration...")
    try:
        data_engine = create_data_engine_from_config(
            env_config=env,
            use_for="historical",
            cache_dir=cache_dir_resolved,  # Override with resolved path
        )
        print(f"[SUCCESS] Data engine created: {type(data_engine).__name__}")
        
        # Check if it's wrapped with cache
        if hasattr(data_engine, '_base_engine'):
            print(f"   Base engine: {type(data_engine._base_engine).__name__}")
            print(f"   Cache enabled: {data_engine._cache_enabled}")
            if hasattr(data_engine, '_cache') and data_engine._cache:
                print(f"   Cache directory: {data_engine._cache._cache_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to create data engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n[INFO] Date Range:")
    print(f"  Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test fetching bars
    print(f"\n[INFO] Fetching bars for {symbol} ({timeframe})...")
    try:
        bars = await data_engine.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
        )
        
        if bars:
            print(f"[SUCCESS] Successfully fetched {len(bars)} bars")
            
            # Show date range
            bar_dates = sorted([b.timestamp for b in bars])
            print(f"   Date range: {bar_dates[0].strftime('%Y-%m-%d %H:%M:%S')} to {bar_dates[-1].strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show sample bars
            print(f"\n[INFO] Sample bars (first 5):")
            for i, bar in enumerate(bars[:5]):
                print(f"   {i+1}. {bar.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,.0f}")
            
            if len(bars) > 5:
                print(f"\n[INFO] Sample bars (last 5):")
                for i, bar in enumerate(bars[-5:]):
                    print(f"   {i+1}. {bar.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                          f"O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:,.0f}")
            
            # Show cache statistics if available
            if hasattr(data_engine, 'get_cache_stats'):
                cache_stats = data_engine.get_cache_stats()
                print(f"\n[INFO] Cache Statistics:")
                print(f"   Total requests: {cache_stats['total_requests']}")
                print(f"   Cache hits: {cache_stats['cache_hits']}")
                print(f"   Cache partial hits: {cache_stats['cache_partial_hits']}")
                print(f"   API calls: {cache_stats['api_calls']}")
                if cache_stats['total_requests'] > 0:
                    hit_rate = (cache_stats['cache_hits'] / cache_stats['total_requests']) * 100
                    print(f"   Cache hit rate: {hit_rate:.1f}%")
        else:
            print(f"[WARNING] No bars returned (empty result)")
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch bars: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test fetching again to verify cache
    print(f"\n[INFO] Fetching again to test cache...")
    try:
        bars2 = await data_engine.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
        )
        
        if bars2:
            print(f"[SUCCESS] Second fetch returned {len(bars2)} bars")
            
            # Check cache statistics again
            if hasattr(data_engine, 'get_cache_stats'):
                cache_stats = data_engine.get_cache_stats()
                print(f"   Cache hits: {cache_stats['cache_hits']}")
                print(f"   API calls: {cache_stats['api_calls']}")
                if cache_stats['total_requests'] > 0:
                    hit_rate = (cache_stats['cache_hits'] / cache_stats['total_requests']) * 100
                    print(f"   Cache hit rate: {hit_rate:.1f}%")
                    
                    if cache_stats['cache_hits'] > 0:
                        print(f"   [SUCCESS] Cache is working! Second request used cached data.")
        else:
            print(f"[WARNING] No bars returned on second fetch")
            
    except Exception as e:
        print(f"[ERROR] Failed on second fetch: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Yahoo Finance data source")
    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Stock symbol to test (default: AAPL)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1D",
        help="Timeframe to test (default: 1D). Options: 1m, 5m, 15m, 30m, 1H, 1D, 1wk, 1mo",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to fetch (default: 30)",
    )
    
    args = parser.parse_args()
    
    await test_yfinance_data_source(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days_back=args.days,
    )


if __name__ == "__main__":
    asyncio.run(main())

