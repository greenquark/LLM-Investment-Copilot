"""Test if dates are marked as 'no data' in cache."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets

async def test():
    env_file = project_root / 'config' / 'env.backtest.yaml'
    env = load_config_with_secrets(env_file)
    data_engine = create_data_engine_from_config(
        env_config=env,
        use_for="historical",
        cache_dir=str(project_root / 'data_cache/bars')
    )
    
    check_date = date(2025, 10, 24)
    day_start = datetime.combine(check_date, datetime.min.time())
    day_end = datetime.combine(check_date, datetime.max.time())
    
    print(f"Checking {check_date}...")
    
    if hasattr(data_engine, '_no_data_ranges'):
        date_key = ('SOXL', '1D', check_date)
        print(f"Date key: {date_key}")
        print(f"In no_data_ranges: {date_key in data_engine._no_data_ranges}")
        if date_key in data_engine._no_data_ranges:
            print(f"  [WARNING] This date is marked as 'no data' in cache!")
            print(f"  The cached engine will skip fetching it.")
    
    print(f"\nAttempting to fetch bars...")
    bars = await data_engine.get_bars('SOXL', day_start, day_end, '1D')
    print(f"Bars returned: {len(bars)}")
    
    if hasattr(data_engine, '_no_data_ranges'):
        date_key = ('SOXL', '1D', check_date)
        print(f"\nAfter fetch, in no_data_ranges: {date_key in data_engine._no_data_ranges}")

if __name__ == "__main__":
    asyncio.run(test())

