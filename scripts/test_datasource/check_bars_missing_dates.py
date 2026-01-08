"""
Script to check why specific dates have no trading data from Yahoo Finance.

This script will:
1. Check if the dates are trading days
2. Try to fetch data from Yahoo Finance for those dates
3. Show what yfinance returns
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets
from core.data import get_trading_days

async def check_dates(symbol: str = "SOXL", dates_to_check: list = None):
    """Check specific dates for missing data."""
    if dates_to_check is None:
        dates_to_check = [
            date(2025, 10, 24),
            date(2025, 11, 7),
            date(2025, 11, 14),
            date(2025, 11, 21),
        ]
    
    print("=" * 80)
    print("CHECKING MISSING DATES FROM YAHOO FINANCE")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Dates to check: {[str(d) for d in dates_to_check]}")
    print()
    
    # Check if these are trading days
    print("Checking if dates are trading days...")
    min_date = min(dates_to_check)
    max_date = max(dates_to_check)
    trading_days = get_trading_days(min_date, max_date)
    
    # Normalize trading days to plain date objects
    trading_dates = []
    for d in trading_days:
        if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
            trading_dates.append(d.date())
        elif isinstance(d, date):
            trading_dates.append(d)
        else:
            try:
                trading_dates.append(date.fromisoformat(str(d)))
            except (ValueError, AttributeError):
                continue
    
    print(f"Trading days in range {min_date} to {max_date}: {len(trading_dates)}")
    print()
    
    for check_date in dates_to_check:
        day_name = check_date.strftime("%A")
        is_trading = check_date in trading_dates
        print(f"{check_date} ({day_name}): {'TRADING DAY' if is_trading else 'NON-TRADING DAY'}")
    
    print()
    print("=" * 80)
    print("ATTEMPTING TO FETCH DATA FROM YAHOO FINANCE")
    print("=" * 80)
    
    # Load configuration
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    
    if not env_file.exists():
        print(f"[ERROR] Config file not found: {env_file}")
        return
    
    env = load_config_with_secrets(env_file)
    
    # Resolve cache directory relative to project root
    data_config = env.get("data", {})
    cache_dir_from_config = data_config.get('cache_dir', 'data_cache/bars')
    cache_dir_path = Path(cache_dir_from_config)
    if not cache_dir_path.is_absolute():
        cache_dir_resolved = str((project_root / cache_dir_from_config).resolve())
    else:
        cache_dir_resolved = str(Path(cache_dir_from_config).resolve())
    
    # Create data engine
    data_engine = create_data_engine_from_config(
        env_config=env,
        use_for="historical",
        cache_dir=cache_dir_resolved,
    )
    
    print(f"Data engine: {type(data_engine).__name__}")
    print()
    
    # Check each date individually
    for check_date in dates_to_check:
        print(f"\n{'=' * 80}")
        print(f"Checking {check_date} ({check_date.strftime('%A')})")
        print(f"{'=' * 80}")
        
        day_start = datetime.combine(check_date, datetime.min.time())
        day_end = datetime.combine(check_date, datetime.max.time())
        
        try:
            bars = await data_engine.get_bars(
                symbol=symbol,
                start=day_start,
                end=day_end,
                timeframe="1D",
            )
            
            if bars:
                print(f"[SUCCESS] Found {len(bars)} bar(s) for {check_date}")
                for bar in bars:
                    print(f"  - {bar.timestamp}: O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, C={bar.close:.2f}, V={bar.volume:,.0f}")
            else:
                print(f"[WARNING] No bars returned for {check_date}")
                print(f"  This could mean:")
                print(f"  1. It's a non-trading day (holiday/weekend)")
                print(f"  2. Data is not yet available from Yahoo Finance (future date)")
                print(f"  3. Yahoo Finance API returned empty data")
                
                # Check if it's marked as "no data" in cache
                if hasattr(data_engine, '_no_data_ranges'):
                    date_key = (symbol.upper(), "1D", check_date)
                    if date_key in data_engine._no_data_ranges:
                        print(f"  [INFO] This date is marked as 'no data' in cache")
        
        except Exception as e:
            print(f"[ERROR] Exception when fetching {check_date}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Note: If dates are in the future (beyond today), Yahoo Finance may not have data yet.")
    print("      Yahoo Finance typically has data up to the most recent trading day.")
    print("      Future dates will return empty data even if they are trading days.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check why specific dates have no data")
    parser.add_argument("--symbol", type=str, default="SOXL", help="Symbol to check")
    parser.add_argument("--dates", type=str, nargs="+", help="Dates to check (YYYY-MM-DD format)")
    
    args = parser.parse_args()
    
    dates_to_check = None
    if args.dates:
        dates_to_check = [date.fromisoformat(d) for d in args.dates]
    
    asyncio.run(check_dates(symbol=args.symbol, dates_to_check=dates_to_check))

