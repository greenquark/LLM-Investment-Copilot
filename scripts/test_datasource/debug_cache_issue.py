"""Debug script to understand why cached engine fails but direct yfinance works."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets
from core.data.yfinance_data import YFinanceDataAdapter

async def test():
    check_date = date(2025, 10, 24)
    day_start = datetime.combine(check_date, datetime.min.time())
    day_end = datetime.combine(check_date, datetime.max.time())
    
    print("=" * 80)
    print("DEBUGGING CACHE ISSUE")
    print("=" * 80)
    print(f"Date: {check_date}")
    print()
    
    # Test 1: Direct yfinance adapter (no cache)
    print("1. Testing direct YFinanceDataAdapter (no cache):")
    print("-" * 80)
    try:
        direct_adapter = YFinanceDataAdapter()
        bars = await direct_adapter.get_bars('SOXL', day_start, day_end, '1D')
        print(f"   Result: {len(bars)} bars")
        if bars:
            for bar in bars:
                print(f"   - {bar.timestamp}: O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, C={bar.close:.2f}")
        else:
            print("   [WARNING] No bars returned from direct adapter!")
    except Exception as e:
        print(f"   [ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 2: Cached engine
    print("2. Testing CachedDataEngine:")
    print("-" * 80)
    try:
        env_file = project_root / 'config' / 'env.backtest.yaml'
        env = load_config_with_secrets(env_file)
        data_engine = create_data_engine_from_config(
            env_config=env,
            use_for="historical",
            cache_dir=str(project_root / 'data_cache/bars')
        )
        
        # Check _no_data_ranges before fetch
        if hasattr(data_engine, '_no_data_ranges'):
            date_key = ('SOXL', '1D', check_date)
            print(f"   _no_data_ranges before fetch: {date_key in data_engine._no_data_ranges}")
            if date_key in data_engine._no_data_ranges:
                print(f"   [WARNING] Date is already marked as 'no data' in _no_data_ranges!")
                print(f"   This will cause the cached engine to skip fetching it.")
        
        # Check cache
        if hasattr(data_engine, '_cache') and data_engine._cache:
            cached_bars = data_engine._cache.get_all_cached_bars('SOXL', '1D')
            if cached_bars:
                cached_dates = {b.timestamp.date() for b in cached_bars}
                print(f"   Cached bars: {len(cached_bars)}")
                print(f"   Date in cache: {check_date in cached_dates}")
                if check_date in cached_dates:
                    matching = [b for b in cached_bars if b.timestamp.date() == check_date]
                    print(f"   Matching bars for date: {len(matching)}")
        
        print(f"\n   Attempting to fetch...")
        bars = await data_engine.get_bars('SOXL', day_start, day_end, '1D')
        print(f"   Result: {len(bars)} bars")
        if bars:
            for bar in bars:
                print(f"   - {bar.timestamp}: O={bar.open:.2f}, H={bar.high:.2f}, L={bar.low:.2f}, C={bar.close:.2f}")
        else:
            print("   [WARNING] No bars returned from cached engine!")
        
        # Check _no_data_ranges after fetch
        if hasattr(data_engine, '_no_data_ranges'):
            date_key = ('SOXL', '1D', check_date)
            print(f"\n   _no_data_ranges after fetch: {date_key in data_engine._no_data_ranges}")
            if date_key in data_engine._no_data_ranges:
                print(f"   [WARNING] Date was marked as 'no data' after fetch!")
        
    except Exception as e:
        print(f"   [ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test())

