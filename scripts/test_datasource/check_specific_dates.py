"""
Check specific dates to see if they're trading days and if data exists.
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
from core.data import is_trading_day


async def check_dates(symbol: str = "SOXL", dates_to_check: list = None):
    """Check if specific dates are trading days and if data exists."""
    if dates_to_check is None:
        dates_to_check = [
            "2024-08-05",  # Monday - has bar
            "2024-08-06",  # Tuesday - should have bar
            "2024-08-07",  # Wednesday - should have bar
            "2024-08-08",  # Thursday - should have bar
            "2024-08-09",  # Friday - should have bar
            "2024-08-10",  # Saturday - weekend
            "2024-08-11",  # Sunday - weekend
            "2024-08-12",  # Monday - has bar
        ]
    
    print("=" * 80)
    print("SPECIFIC DATE CHECK")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print("=" * 80)
    print()
    
    # Load config
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    env = load_config_with_secrets(env_file)
    
    # Create data engine
    data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
    
    for date_str in dates_to_check:
        check_date = date.fromisoformat(date_str)
        print(f"\n{date_str} ({check_date.strftime('%A')}):")
        
        # Check if it's a trading day
        is_trading = is_trading_day(check_date)
        print(f"  Trading Day: {is_trading}")
        
        if is_trading:
            # Try to fetch data for this date
            day_start = datetime.combine(check_date, datetime.min.time())
            day_end = datetime.combine(check_date, datetime.max.time())
            
            try:
                bars = await data_engine.get_bars(symbol, day_start, day_end, "1D")
                if bars:
                    bar = bars[0]
                    print(f"  Data: YES - O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, C={bar.close:.2f}, V={bar.volume:,.0f}")
                else:
                    print(f"  Data: NO - No bars returned")
            except Exception as e:
                print(f"  Data: ERROR - {e}")
        else:
            print(f"  Data: N/A (not a trading day)")
    
    print("\n" + "=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(check_dates())
