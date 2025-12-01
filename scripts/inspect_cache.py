"""
Simple script to inspect cache contents.

This script shows what data is currently cached, including:
- Which symbols and timeframes are cached
- Date ranges for each cache
- Number of bars in each cache
- Sample timestamps

Usage:
    python scripts/inspect_cache.py
    python scripts/inspect_cache.py --symbol SOXL --timeframe 1D
    python scripts/inspect_cache.py --cache-dir data_cache/bars
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.data.cache import DataCache
    from core.models.bar import Bar
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


async def inspect_cache(
    cache_dir: str = "data_cache/bars",
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> None:
    """
    Inspect cache contents.
    
    Args:
        cache_dir: Directory containing cache files
        symbol: Optional symbol to filter by (e.g., "SOXL")
        timeframe: Optional timeframe to filter by (e.g., "1D")
    """
    cache = DataCache(cache_dir)
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory does not exist: {cache_dir}")
        return
    
    print("=" * 80)
    print("Cache Inspection")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}\n")
    
    # Find all cache files
    cache_files = list(cache_path.glob("*.parquet"))
    
    if not cache_files:
        print("No cache files found.")
        return
    
    print(f"Found {len(cache_files)} cache file(s):\n")
    
    # Parse cache files
    cache_info = []
    for cache_file in sorted(cache_files):
        # Parse filename: SYMBOL_TIMEFRAME.parquet
        filename = cache_file.stem
        parts = filename.rsplit("_", 1)
        if len(parts) != 2:
            print(f"⚠️  Skipping file with unexpected format: {filename}")
            continue
        
        file_symbol = parts[0]
        file_timeframe = parts[1]
        
        # Apply filters
        if symbol and file_symbol.upper() != symbol.upper():
            continue
        if timeframe and file_timeframe.upper() != cache._normalize_timeframe(timeframe).upper():
            continue
        
        try:
            # Load all bars from this cache file
            all_bars = cache.get_all_cached_bars(file_symbol, file_timeframe)
            
            if all_bars:
                timestamps = sorted([b.timestamp for b in all_bars])
                min_date = timestamps[0].date()
                max_date = timestamps[-1].date()
                num_bars = len(all_bars)
                
                cache_info.append({
                    "symbol": file_symbol,
                    "timeframe": file_timeframe,
                    "file": cache_file,
                    "num_bars": num_bars,
                    "min_date": min_date,
                    "max_date": max_date,
                    "min_timestamp": timestamps[0],
                    "max_timestamp": timestamps[-1],
                    "bars": all_bars,
                })
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")
            continue
    
    if not cache_info:
        if symbol or timeframe:
            print(f"No cache files found matching symbol={symbol}, timeframe={timeframe}")
        else:
            print("No valid cache files found.")
        return
    
    # Display cache information
    for i, info in enumerate(cache_info, 1):
        print(f"{i}. {info['symbol']} ({info['timeframe']})")
        print(f"   File: {info['file'].name}")
        print(f"   Bars: {info['num_bars']:,}")
        print(f"   Date range: {info['min_date']} to {info['max_date']}")
        print(f"   Timestamp range: {info['min_timestamp']} to {info['max_timestamp']}")
        
        # Show sample bars
        if info['num_bars'] > 0:
            sample_bars = info['bars'][:30]
            print(f"   Sample bars:")
            for bar in sample_bars:
                print(f"     - {bar.timestamp}: O={bar.open:.2f}, H={bar.high:.2f}, "
                      f"L={bar.low:.2f}, C={bar.close:.2f}, V={bar.volume:,.0f}")
            if info['num_bars'] > 30:
                print(f"     ... and {info['num_bars'] - 30} more bars")
        
        # Check for gaps (for daily bars)
        if info['timeframe'].upper().endswith("D") or info['timeframe'].upper() == "D":
            try:
                from core.data import get_trading_days
                
                trading_days = get_trading_days(info['min_date'], info['max_date'])
                
                # Normalize everything to plain date objects
                # Note: pandas Timestamp is a subclass of date, so isinstance(d, date) returns True
                # We need to check if it has a date() method (Timestamp/datetime) vs plain date
                trading_dates = []
                for d in trading_days:
                    # Check if it has a date() method (pandas Timestamp or datetime objects)
                    # Plain Python date objects don't have a date() method
                    if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                        trading_dates.append(d.date())
                    elif isinstance(d, date):
                        trading_dates.append(d)
                    else:
                        try:
                            trading_dates.append(date.fromisoformat(str(d)))
                        except (ValueError, AttributeError):
                            continue
                cached_dates = {
                    b.timestamp.date() if hasattr(b.timestamp, "date") else b.timestamp
                    for b in info['bars']
                }
                
                missing_dates = [d for d in trading_dates if d not in cached_dates]
                
                if missing_dates:
                    print(
                        f"   ⚠️  Missing {len(missing_dates)} trading day(s): "
                        f"{missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                    )
                else:
                    print(
                        f"   ✓ All {len(trading_dates)} trading days in range are cached"
                    )
            except ImportError:
                pass  # Trading calendar not available
        
        print()
    
    # Summary
    total_bars = sum(info['num_bars'] for info in cache_info)
    total_size = sum(info['file'].stat().st_size for info in cache_info)
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total cache files: {len(cache_info)}")
    print(f"Total bars cached: {total_bars:,}")
    print(f"Total cache size: {total_size / 1024 / 1024:.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect cache contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all cached data
  python scripts/inspect_cache.py
  
  # Show only SOXL daily bars
  python scripts/inspect_cache.py --symbol SOXL --timeframe 1D
  
  # Use custom cache directory
  python scripts/inspect_cache.py --cache-dir my_cache/bars
        """
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data_cache/bars",
        help="Cache directory path (default: data_cache/bars)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Filter by symbol (e.g., SOXL, TQQQ)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Filter by timeframe (e.g., 1D, 15m, 1H)",
    )
    
    args = parser.parse_args()
    
    asyncio.run(inspect_cache(
        cache_dir=args.cache_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
    ))


if __name__ == "__main__":
    main()

