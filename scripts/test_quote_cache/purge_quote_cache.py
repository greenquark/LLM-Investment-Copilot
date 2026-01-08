"""
Script to purge cache files for specific tickers.

This script:
1. Lists all cached tickers with their date ranges
2. Prompts the user to select which ticker to purge
3. Deletes the cache file(s) for the selected ticker

Usage:
    python scripts/test_quote_cache/purge_quote_cache.py
    python scripts/test_quote_cache/purge_quote_cache.py --cache-dir data_cache/bars
"""

import sys
from pathlib import Path
from datetime import date
from typing import Optional, List, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.data.cache import DataCache
    from core.models.bar import Bar
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


def list_cached_tickers(cache_dir: str = "data_cache/bars") -> List[Dict]:
    """
    List all cached tickers with their date ranges.
    
    Args:
        cache_dir: Directory containing cache files (relative to project root or absolute)
        
    Returns:
        List of dictionaries with cache information
    """
    # Resolve cache directory relative to project root if it's a relative path
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        # Make it relative to project root
        cache_path = project_root / cache_dir
    
    if not cache_path.exists():
        print(f"[ERROR] Cache directory does not exist: {cache_path}")
        print(f"   (resolved from: {cache_dir})")
        print(f"   Project root: {project_root}")
        return []
    
    # Use the resolved path for DataCache
    cache = DataCache(str(cache_path))
    
    # Find all cache files
    cache_files = list(cache_path.glob("*.parquet"))
    
    if not cache_files:
        print("No cache files found.")
        return []
    
    # Parse cache files
    cache_info = []
    for cache_file in sorted(cache_files):
        # Parse filename: SYMBOL_TIMEFRAME.parquet
        filename = cache_file.stem
        parts = filename.rsplit("_", 1)
        if len(parts) != 2:
            print(f"[WARNING] Skipping file with unexpected format: {filename}")
            continue
        
        file_symbol = parts[0]
        file_timeframe = parts[1]
        
        try:
            # Load all bars from this cache file
            all_bars = cache.get_all_cached_bars(file_symbol, file_timeframe)
            
            if all_bars:
                timestamps = sorted([b.timestamp for b in all_bars])
                min_date = timestamps[0].date()
                max_date = timestamps[-1].date()
                num_bars = len(all_bars)
                file_size = cache_file.stat().st_size
                
                # Calculate missing dates for daily bars
                missing_dates = []
                if file_timeframe.upper().endswith("D") or file_timeframe.upper() == "D":
                    try:
                        from core.data import get_trading_days
                        
                        trading_days = get_trading_days(min_date, max_date)
                        
                        # Normalize everything to plain date objects
                        trading_dates = []
                        for d in trading_days:
                            # Check if it has a date() method (pandas Timestamp or datetime objects)
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
                            for b in all_bars
                        }
                        
                        missing_dates = [d for d in trading_dates if d not in cached_dates]
                    except ImportError:
                        pass  # Trading calendar not available
                
                cache_info.append({
                    "symbol": file_symbol,
                    "timeframe": file_timeframe,
                    "file": cache_file,
                    "num_bars": num_bars,
                    "min_date": min_date,
                    "max_date": max_date,
                    "file_size": file_size,
                    "missing_dates": missing_dates,
                    "bars": all_bars,  # Keep bars for date calculation
                })
        except Exception as e:
            print(f"[WARNING] Error loading {filename}: {e}")
            continue
    
    return cache_info


def display_cache_list(cache_info: List[Dict]) -> None:
    """Display the list of cached tickers with missing dates."""
    if not cache_info:
        print("No cached data found.")
        return
    
    print("=" * 80)
    print("Cached Tickers")
    print("=" * 80)
    print()
    
    for i, info in enumerate(cache_info, 1):
        file_size_mb = info['file_size'] / 1024 / 1024
        print(f"{i}. {info['symbol']} ({info['timeframe']})")
        print(f"   Date range: {info['min_date']} to {info['max_date']}")
        print(f"   Bars: {info['num_bars']:,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   File: {info['file'].name}")
        
        # Display missing dates
        missing_dates = info.get('missing_dates', [])
        if missing_dates:
            print(f"   [WARNING] Missing {len(missing_dates)} trading day(s)")
            # Show first 10 missing dates, then summarize if more
            if len(missing_dates) <= 10:
                print(f"   Missing dates: {', '.join(str(d) for d in missing_dates)}")
            else:
                print(f"   Missing dates (first 10): {', '.join(str(d) for d in missing_dates[:10])}")
                print(f"   ... and {len(missing_dates) - 10} more missing dates")
        elif info['timeframe'].upper().endswith("D") or info['timeframe'].upper() == "D":
            # For daily bars, if no missing dates, show success message
            try:
                from core.data import get_trading_days
                trading_days = get_trading_days(info['min_date'], info['max_date'])
                # Normalize trading days count
                trading_dates_count = 0
                for d in trading_days:
                    if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                        trading_dates_count += 1
                    elif isinstance(d, date):
                        trading_dates_count += 1
                    else:
                        try:
                            date.fromisoformat(str(d))
                            trading_dates_count += 1
                        except (ValueError, AttributeError):
                            pass
                print(f"   [OK] All {trading_dates_count} trading days in range are cached")
            except ImportError:
                pass  # Trading calendar not available
        
        print()


def purge_ticker(cache_info: List[Dict], selection: int) -> bool:
    """
    Purge cache for the selected ticker.
    
    Args:
        cache_info: List of cache information dictionaries
        selection: Index of the ticker to purge (1-based)
        
    Returns:
        True if successful, False otherwise
    """
    if selection < 1 or selection > len(cache_info):
        print(f"[ERROR] Invalid selection: {selection}")
        return False
    
    selected = cache_info[selection - 1]
    cache_file = selected['file']
    symbol = selected['symbol']
    timeframe = selected['timeframe']
    
    print()
    print("=" * 80)
    print(f"Purging cache for {symbol} ({timeframe})")
    print("=" * 80)
    print(f"File: {cache_file}")
    print(f"Bars: {selected['num_bars']:,}")
    print(f"Date range: {selected['min_date']} to {selected['max_date']}")
    
    # Show missing dates if any
    missing_dates = selected.get('missing_dates', [])
    if missing_dates:
        print(f"Missing dates: {len(missing_dates)} trading day(s)")
        if len(missing_dates) <= 20:
            print(f"   {', '.join(str(d) for d in missing_dates)}")
        else:
            print(f"   {', '.join(str(d) for d in missing_dates[:20])}")
            print(f"   ... and {len(missing_dates) - 20} more")
    print()
    
    # Confirm deletion
    confirm = input(f"Are you sure you want to delete this cache file? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("Cancelled. No files were deleted.")
        return False
    
    try:
        # Delete the cache file
        cache_file.unlink()
        print(f"[SUCCESS] Deleted cache file: {cache_file.name}")
        
        # Also clear in-memory cache if possible
        try:
            cache = DataCache(str(cache_file.parent))
            # The in-memory cache will be cleared on next access since file doesn't exist
            print(f"[SUCCESS] Cache purged for {symbol} ({timeframe})")
        except Exception as e:
            print(f"[WARNING] Could not clear in-memory cache: {e}")
            print(f"[INFO] File deleted successfully, but in-memory cache may still exist")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete cache file: {e}")
        return False


def main():
    """Main function to run the purge script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Purge cache files for specific tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - list and select ticker to purge
  python scripts/test_quote_cache/purge_quote_cache.py
  
  # Use custom cache directory
  python scripts/test_quote_cache/purge_quote_cache.py --cache-dir my_cache/bars
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
        help="Directly purge a specific symbol (skips interactive selection)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help="Timeframe to purge (required if --symbol is specified)",
    )
    
    args = parser.parse_args()
    
    # List cached tickers
    cache_info = list_cached_tickers(args.cache_dir)
    
    if not cache_info:
        print("No cached data found. Nothing to purge.")
        return
    
    # If symbol is specified, purge directly
    if args.symbol:
        if not args.timeframe:
            print("[ERROR] --timeframe is required when using --symbol")
            return
        
        # Find matching cache entry
        matching = None
        for info in cache_info:
            if info['symbol'].upper() == args.symbol.upper() and \
               info['timeframe'].upper() == args.timeframe.upper():
                matching = info
                break
        
        if not matching:
            print(f"[ERROR] No cache found for {args.symbol} ({args.timeframe})")
            return
        
        # Purge directly
        selection = cache_info.index(matching) + 1
        purge_ticker(cache_info, selection)
        return
    
    # Interactive mode: loop until user quits
    while True:
        # Refresh cache list (in case files were deleted)
        cache_info = list_cached_tickers(args.cache_dir)
        
        if not cache_info:
            print("No cached data found. Nothing to purge.")
            print()
            user_input = input("Press Enter to refresh, or 'q' to quit: ").strip().lower()
            if user_input in ['q', 'quit', 'exit']:
                print("Exiting.")
                return
            continue
        
        # Display list
        display_cache_list(cache_info)
        
        print("=" * 80)
        print("Select ticker to purge")
        print("=" * 80)
        print("Enter the number (1-{}) of the ticker to purge, or 'q' to quit:".format(len(cache_info)))
        
        try:
            user_input = input("Selection: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting.")
                return
            
            selection = int(user_input)
            
            if purge_ticker(cache_info, selection):
                print()
                print("=" * 80)
                print("Purge completed successfully!")
                print("=" * 80)
                print()
                # Ask if user wants to continue
                continue_input = input("Purge another ticker? (y/n, or 'q' to quit): ").strip().lower()
                if continue_input in ['q', 'quit', 'exit', 'n', 'no']:
                    print("Exiting.")
                    return
                # Continue loop to refresh cache list
            else:
                # Purge was cancelled or failed
                print()
                continue_input = input("Try again? (y/n, or 'q' to quit): ").strip().lower()
                if continue_input in ['q', 'quit', 'exit', 'n', 'no']:
                    print("Exiting.")
                    return
                # Continue loop
                
        except ValueError:
            print(f"[ERROR] Invalid input. Please enter a number between 1 and {len(cache_info)}, or 'q' to quit.")
            print()
            continue
        except KeyboardInterrupt:
            print("\n\nExiting.")
            return
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            print()
            continue_input = input("Continue? (y/n, or 'q' to quit): ").strip().lower()
            if continue_input in ['q', 'quit', 'exit', 'n', 'no']:
                print("Exiting.")
                return
            continue


if __name__ == "__main__":
    main()

