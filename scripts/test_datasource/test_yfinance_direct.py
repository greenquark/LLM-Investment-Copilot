"""Test yfinance directly to see what it returns for specific dates."""

import yfinance as yf
import pandas as pd
from datetime import date, datetime

dates_to_check = [
    date(2025, 10, 24),
    date(2025, 11, 7),
    date(2025, 11, 14),
    date(2025, 11, 21),
]

symbol = "SOXL"

print("=" * 80)
print("TESTING YFINANCE DIRECTLY")
print("=" * 80)
print(f"Symbol: {symbol}")
print()

for check_date in dates_to_check:
    print(f"\n{'=' * 80}")
    print(f"Testing {check_date} ({check_date.strftime('%A')})")
    print(f"{'=' * 80}")
    
    # Test 1: Using yf.download (what our code uses)
    print("\n1. Using yf.download() (what our code uses):")
    try:
        df_download = yf.download(
            tickers=symbol,
            start=check_date,
            end=date(check_date.year, check_date.month, check_date.day + 1),
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        print(f"   Shape: {df_download.shape}")
        print(f"   Empty: {df_download.empty}")
        print(f"   Type: {type(df_download)}")
        print(f"   Columns type: {type(df_download.columns)}")
        if isinstance(df_download.columns, pd.MultiIndex):
            print(f"   MultiIndex columns: {df_download.columns.nlevels} levels")
            print(f"   Level 0: {list(df_download.columns.get_level_values(0))[:5]}")
            print(f"   Level 1: {list(df_download.columns.get_level_values(1))[:5]}")
        else:
            print(f"   Columns: {list(df_download.columns)[:10]}")
        if not df_download.empty:
            print(f"   Index (first 3): {list(df_download.index[:3])}")
            print(f"   First row date: {df_download.index[0]}")
            if isinstance(df_download.index[0], pd.Timestamp):
                print(f"   First row date (Python): {df_download.index[0].to_pydatetime()}")
                print(f"   First row date (date only): {df_download.index[0].date()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Using ticker.history (alternative method)
    print("\n2. Using ticker.history() (alternative method):")
    try:
        ticker = yf.Ticker(symbol)
        df_history = ticker.history(
            start=check_date,
            end=date(check_date.year, check_date.month, check_date.day + 1),
            interval="1d",
        )
        print(f"   Shape: {df_history.shape}")
        print(f"   Empty: {df_history.empty}")
        print(f"   Type: {type(df_history)}")
        print(f"   Columns: {list(df_history.columns)[:10]}")
        if not df_history.empty:
            print(f"   Index (first 3): {list(df_history.index[:3])}")
            print(f"   First row date: {df_history.index[0]}")
            if isinstance(df_history.index[0], pd.Timestamp):
                print(f"   First row date (Python): {df_history.index[0].to_pydatetime()}")
                print(f"   First row date (date only): {df_history.index[0].date()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

