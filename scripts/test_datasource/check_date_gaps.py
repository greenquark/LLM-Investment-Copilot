"""
Diagnostic script to check for date gaps in bar data for a specific date range.
This helps troubleshoot why charts show missing dates.
"""
import asyncio
from datetime import datetime, date
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets
from core.data import get_trading_days, is_trading_day


async def check_date_gaps(symbol: str = "SOXL", start_date: str = "2024-07-05", end_date: str = "2025-12-15"):
    """Check for missing dates in bar data."""
    print("=" * 80)
    print("DATE GAP DIAGNOSTIC")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} to {end_date}")
    print("=" * 80)
    print()
    
    # Load config
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    env = load_config_with_secrets(env_file)
    
    # Create data engine
    data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
    
    # Parse dates
    start = datetime.fromisoformat(f"{start_date}T00:00:00")
    end = datetime.fromisoformat(f"{end_date}T23:59:59")
    
    print(f"Fetching bars from {start.date()} to {end.date()}...")
    bars = await data_engine.get_bars(symbol, start, end, "1D")
    
    print(f"\nRetrieved {len(bars)} bars")
    if not bars:
        print("ERROR: No bars returned!")
        return
    
    # Get all bar dates
    bar_dates = sorted([b.timestamp.date() for b in bars])
    print(f"Bar date range: {bar_dates[0]} to {bar_dates[-1]}")
    print()
    
    # Get all trading days in the range
    print("Checking trading days...")
    try:
        trading_days = get_trading_days(bar_dates[0], bar_dates[-1])
        print(f"Total trading days in range: {len(trading_days)}")
        
        # Find missing trading days
        bar_dates_set = set(bar_dates)
        missing_trading_days = [d for d in trading_days if d not in bar_dates_set]
        
        if missing_trading_days:
            print(f"\n[WARNING] Found {len(missing_trading_days)} missing trading days:")
            print()
            
            # Group consecutive missing days
            gaps = []
            current_gap_start = None
            current_gap_end = None
            
            for missing_day in sorted(missing_trading_days):
                if current_gap_start is None:
                    current_gap_start = missing_day
                    current_gap_end = missing_day
                elif (missing_day - current_gap_end).days == 1:
                    # Consecutive day
                    current_gap_end = missing_day
                else:
                    # Gap ended, save it
                    gaps.append((current_gap_start, current_gap_end))
                    current_gap_start = missing_day
                    current_gap_end = missing_day
            
            # Add last gap
            if current_gap_start is not None:
                gaps.append((current_gap_start, current_gap_end))
            
            # Print gaps
            for gap_start, gap_end in gaps:
                if gap_start == gap_end:
                    print(f"  Missing: {gap_start} ({gap_start.strftime('%A')})")
                else:
                    print(f"  Missing: {gap_start} to {gap_end} ({len([d for d in trading_days if gap_start <= d <= gap_end])} trading days)")
            
            # Focus on July-Nov 2024
            print("\n" + "=" * 80)
            print("JULY - NOVEMBER 2024 ANALYSIS")
            print("=" * 80)
            july_nov_start = date(2024, 7, 1)
            july_nov_end = date(2024, 11, 30)
            july_nov_missing = [d for d in missing_trading_days if july_nov_start <= d <= july_nov_end]
            
            if july_nov_missing:
                print(f"Found {len(july_nov_missing)} missing trading days in July-Nov 2024:")
                for d in july_nov_missing:
                    print(f"  {d} ({d.strftime('%A')})")
            else:
                print("No missing trading days in July-Nov 2024")
            
        else:
            print("\n[OK] No missing trading days found!")
            print("All trading days in the range have data.")
        
        # Check for non-trading days in bar dates (shouldn't happen for daily bars)
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECK")
        print("=" * 80)
        non_trading_in_bars = []
        for bar_date in bar_dates:
            if not is_trading_day(bar_date):
                non_trading_in_bars.append(bar_date)
        
        if non_trading_in_bars:
            print(f"[WARNING] Found {len(non_trading_in_bars)} non-trading days in bar data:")
            for d in non_trading_in_bars[:10]:  # Show first 10
                print(f"  {d} ({d.strftime('%A')})")
            if len(non_trading_in_bars) > 10:
                print(f"  ... and {len(non_trading_in_bars) - 10} more")
        else:
            print("[OK] All bar dates are trading days (as expected)")
        
    except Exception as e:
        print(f"\n[WARNING] Could not check trading days: {e}")
        print("Falling back to simple date gap detection...")
        
        # Simple gap detection without trading calendar
        print(f"\nBar dates: {len(bar_dates)}")
        print(f"Date range: {bar_dates[0]} to {bar_dates[-1]}")
        
        # Find gaps (consecutive dates that are missing)
        all_dates = set(bar_dates)
        gaps = []
        current_date = bar_dates[0]
        while current_date <= bar_dates[-1]:
            if current_date not in all_dates:
                gap_start = current_date
                while current_date not in all_dates and current_date <= bar_dates[-1]:
                    current_date = date(current_date.year, current_date.month, current_date.day + 1) if current_date.day < 28 else date(current_date.year, current_date.month + 1, 1)
                gap_end = date(current_date.year, current_date.month, current_date.day - 1)
                gaps.append((gap_start, gap_end))
            else:
                current_date = date(current_date.year, current_date.month, current_date.day + 1) if current_date.day < 28 else date(current_date.year, current_date.month + 1, 1)
        
        if gaps:
            print(f"\nFound {len(gaps)} date gaps:")
            for gap_start, gap_end in gaps:
                print(f"  {gap_start} to {gap_end}")
        else:
            print("\nNo date gaps found")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check for date gaps in bar data")
    parser.add_argument("--symbol", default="SOXL", help="Symbol to check")
    parser.add_argument("--start", default="2024-07-05", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-15", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    asyncio.run(check_date_gaps(args.symbol, args.start, args.end))
