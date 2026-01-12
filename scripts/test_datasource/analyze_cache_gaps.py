"""
Analyze cache to identify why date gaps appear in charts.
This script checks what dates are actually in the cache file vs what should be there.
"""
import asyncio
from datetime import datetime, date
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets
from core.data import get_trading_days, is_trading_day


async def analyze_cache(symbol: str = "SOXL", timeframe: str = "1D"):
    """Analyze cache file to see what dates are actually stored."""
    print("=" * 80)
    print("CACHE ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print("=" * 80)
    print()
    
    # Load config
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    env = load_config_with_secrets(env_file)
    
    # Get cache directory
    data_config = env.get("data", {})
    cache_dir_from_config = data_config.get('cache_dir', 'data_cache/bars')
    cache_dir_path = Path(cache_dir_from_config)
    if not cache_dir_path.is_absolute():
        cache_dir_resolved = project_root / cache_dir_from_config
    else:
        cache_dir_resolved = Path(cache_dir_from_config)
    
    # Find cache file
    cache_file = cache_dir_resolved / f"{symbol.upper()}_{timeframe}.parquet"
    
    print(f"Cache directory: {cache_dir_resolved}")
    print(f"Cache file: {cache_file}")
    print()
    
    if not cache_file.exists():
        print("ERROR: Cache file does not exist!")
        return
    
    # Read cache file
    try:
        df = pd.read_parquet(cache_file)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        print(f"Cache file contains {len(df)} bars")
        print(f"Date range in cache: {df.index[0].date()} to {df.index[-1].date()}")
        print()
        
        # Check Aug 5-12, 2024 specifically
        print("=" * 80)
        print("AUGUST 5-12, 2024 ANALYSIS")
        print("=" * 80)
        
        aug_start = date(2024, 8, 5)
        aug_end = date(2024, 8, 12)
        
        # Get all dates in cache for this range
        cache_dates = [d.date() for d in df.index if aug_start <= d.date() <= aug_end]
        cache_dates_set = set(cache_dates)
        
        print(f"Dates in cache (Aug 5-12): {sorted(cache_dates)}")
        print(f"Count: {len(cache_dates)}")
        print()
        
        # Get expected trading days
        expected_dates = get_trading_days(aug_start, aug_end)
        expected_dates_set = set(expected_dates)
        
        print(f"Expected trading days (Aug 5-12): {sorted(expected_dates)}")
        print(f"Count: {len(expected_dates)}")
        print()
        
        # Find missing dates
        missing = expected_dates_set - cache_dates_set
        if missing:
            print(f"WARNING: Missing dates in cache: {sorted(missing)}")
        else:
            print("OK: All expected trading days are in cache")
        print()
        
        # Check for duplicate timestamps
        duplicates = df.index[df.index.duplicated(keep=False)]
        if len(duplicates) > 0:
            print(f"WARNING: Found {len(duplicates)} duplicate timestamps in cache!")
            print(f"Duplicate timestamps: {sorted(set(duplicates))[:10]}")
        else:
            print("OK: No duplicate timestamps")
        print()
        
        # Check for gaps in the entire date range
        print("=" * 80)
        print("CHECKING FOR GAPS IN ENTIRE CACHE RANGE")
        print("=" * 80)
        
        cache_start = df.index[0].date()
        cache_end = df.index[-1].date()
        
        # Get all trading days in cache range
        all_trading_days = get_trading_days(cache_start, cache_end)
        all_cache_dates = set(d.date() for d in df.index)
        
        missing_in_range = set(all_trading_days) - all_cache_dates
        
        if missing_in_range:
            print(f"WARNING: Found {len(missing_in_range)} missing trading days in cache range")
            # Show first 20 missing dates
            missing_sorted = sorted(missing_in_range)
            print(f"First 20 missing dates: {missing_sorted[:20]}")
            
            # Check if Aug 5-12 dates are in the missing list
            aug_missing = missing_in_range & expected_dates_set
            if aug_missing:
                print(f"\nCRITICAL: Aug 5-12 dates missing from cache: {sorted(aug_missing)}")
        else:
            print("OK: All trading days in cache range are present")
        
    except Exception as e:
        print(f"ERROR reading cache file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(analyze_cache())
