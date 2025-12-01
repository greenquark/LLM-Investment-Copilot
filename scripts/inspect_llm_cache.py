"""
Script to inspect LLM trend detection cache state.

This script shows what's currently in the in-memory cache registry.
Note: The cache is in-memory only and doesn't persist across runs.
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models.llm_trend import get_all_states, get_state


def inspect_cache():
    """Inspect the current state of the LLM trend cache."""
    print("=" * 60)
    print("LLM Trend Detection Cache Inspection")
    print("=" * 60)
    
    all_states = get_all_states()
    
    if not all_states:
        print("\n❌ Cache is empty (no states cached)")
        print("\nNote: Cache is file-based and persists across runs.")
        print("      Cache location: data_cache/llm_trend/")
        return
    
    print(f"\n✓ Found {len(all_states)} cached state(s):\n")
    
    for (symbol, timeframe, state_date), state in all_states.items():
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Date: {state_date} ({state.as_of})")
        print(f"Regime: {state.regime_final}")
        print(f"Trend Strength: {state.trend_strength:.2f}")
        print(f"Range Strength: {state.range_strength:.2f}")
        if state.summary_for_user:
            print(f"Summary: {state.summary_for_user[:100]}...")
        print("-" * 60)
    
    # Check for specific date
    print("\n" + "=" * 60)
    print("Checking for specific dates:")
    print("=" * 60)
    
    test_symbol = "SOXL"
    test_timeframe = "1D"
    test_date = datetime(2025, 7, 15).date()
    
    # Check for specific date
    state = get_state(test_symbol, test_timeframe, as_of_date=test_date)
    if state:
        state_date = state.as_of.date()
        print(f"\n✓ Found cached state for {test_symbol} ({test_timeframe}) on {test_date}:")
        print(f"  Cached date: {state_date}")
        print(f"  Requested date: {test_date}")
        if state_date == test_date:
            print(f"  ✓ Date matches! Cache hit for {test_date}")
        else:
            print(f"  ✗ Date mismatch! Cache miss for {test_date}")
            print(f"    (Cache has {state_date}, but need {test_date})")
    else:
        print(f"\n✗ No cached state found for {test_symbol} ({test_timeframe}) on {test_date}")
    
    # Also check latest state
    latest_state = get_state(test_symbol, test_timeframe)
    if latest_state:
        print(f"\n  Latest cached state: {latest_state.as_of.date()}")
    
    print("\n" + "=" * 60)
    print("Cache Information:")
    print("=" * 60)
    print("1. File-based persistence - cache persists across script runs")
    print("2. Stores multiple dates per (symbol, timeframe) pair")
    print("3. Cache location: data_cache/llm_trend/")
    print("4. Cache key: (symbol, timeframe, date)")
    print("5. Format: JSON files (one per symbol/timeframe)")


if __name__ == "__main__":
    inspect_cache()

