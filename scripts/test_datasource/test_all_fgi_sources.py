"""
Comprehensive test script for all Fear & Greed Index data sources.

This script:
1. Tests all available FGI data sources (CSV files, CNN API, QuantConnect, etc.)
2. Samples dates to see which sources have values from when
3. Compares values across all sources
4. Prints comprehensive comparison table
5. Shows data availability by source and date range

Data Sources:
- fear-greed-2011-2023.csv (fallback CSV with Date, Fear Greed)
- CNN API (production.dataviz.cnn.io)
- QuantConnect API (if available)

Note: fear_greed_index.csv has been removed as a data source due to corrupted data.

12/30/2025 test result:
CACHE STATISTICS:
  Total records: 3748
  Date range: 2011-01-03 to 2025-12-29
  Calendar days span: 5475 days
  Trading days in range: 3770 days
  Coverage (calendar days): 68.5%
  Coverage (trading days): 99.4%

  Records with classification: 1372 (36.6%)

GAPS IN CACHE (5 gap(s), 22 trading days missing):
  2020-06-08 to 2020-06-12 (5 calendar days, 5 trading days)
  2020-06-15 to 2020-06-19 (5 calendar days, 5 trading days)
  2020-06-22 to 2020-06-26 (5 calendar days, 5 trading days)
  2020-06-29 to 2020-07-02 (4 calendar days, 4 trading days)
  2020-07-06 to 2020-07-08 (3 calendar days, 3 trading days)


"""

import sys
import random
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Tuple, List, Optional
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    print("[WARN] pandas not available")

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    print("[WARN] httpx not available")

from core.data.fear_greed_index import (
    _get_historical_fgi_from_api,
    _get_historical_fgi_from_fallback_csv,
    _load_fallback_csv,
    initialize_cache,
    _get_cache_path,
    _load_from_cache,
    _save_to_cache,
)
from core.data import is_trading_day


def load_fallback_csv() -> Optional[Dict[date, Tuple[float, Optional[str]]]]:
    """
    Load fallback CSV file (fear-greed-2011-2023.csv).
    
    Returns:
        Dictionary mapping date -> (value, None) or None
    """
    fallback_data = _load_fallback_csv()
    return fallback_data


def fetch_cnn_api(target_date: date) -> Optional[Tuple[float, Optional[str]]]:
    """Fetch from CNN API."""
    return _get_historical_fgi_from_api(target_date)


def fetch_quantconnect_api(
    target_date: date,
    api_key: Optional[str] = None,
) -> Optional[Tuple[float, Optional[str]]]:
    """
    Attempt to fetch from QuantConnect API.
    
    Returns:
        Tuple of (value, classification) or None
    """
    if not _HTTPX_AVAILABLE:
        return None
    
    date_str = target_date.strftime("%Y-%m-%d")
    base_url = "https://www.quantconnect.com/api/v2"
    
    endpoints = [
        f"{base_url}/data/fear-and-greed",
        f"{base_url}/data/feargreed",
        f"{base_url}/datasets/fear-and-greed",
    ]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    for endpoint in endpoints:
        try:
            url = f"{endpoint}?date={date_str}"
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        if "value" in data:
                            value = float(data["value"])
                            classification = data.get("classification") or data.get("rating")
                            return value, classification
        except Exception:
            continue
    
    return None


def check_cache_range() -> None:
    """
    Check and display the date range and statistics of the FGI cache.
    """
    if not _PANDAS_AVAILABLE:
        print("[ERROR] pandas not available, cannot check cache")
        return
    
    # Initialize cache if needed
    initialize_cache()
    
    cache_path = _get_cache_path()
    
    print("=" * 100)
    print("FGI CACHE INFORMATION")
    print("=" * 100)
    print()
    
    if cache_path is None:
        print("[INFO] Cache is disabled")
        return
    
    if not cache_path.exists():
        print("[INFO] Cache file does not exist")
        print(f"   Cache path: {cache_path}")
        return
    
    try:
        # Load cache
        df = pd.read_parquet(cache_path)
        
        # Ensure date column is date type
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            dates = df['date'].tolist()
        else:
            # Date is index
            if df.index.name == 'date':
                df.index = pd.to_datetime(df.index).date
            dates = df.index.tolist()
        
        if not dates:
            print("[INFO] Cache file exists but is empty")
            print(f"   Cache path: {cache_path}")
            return
        
        dates_sorted = sorted(dates)
        min_date = dates_sorted[0]
        max_date = dates_sorted[-1]
        count = len(dates_sorted)
        
        # Check for gaps (only trading days without data)
        gaps = []
        non_trading_days_in_range = []
        current = min_date
        gap_start = None
        while current <= max_date:
            is_trading = True
            try:
                is_trading = is_trading_day(current)
            except Exception:
                # If trading day check fails, assume it's a trading day
                pass
            
            if current not in dates_sorted:
                if is_trading:
                    # Trading day without data - potential gap
                    if gap_start is None:
                        gap_start = current
                else:
                    # Non-trading day - not a gap, but track it
                    if gap_start is not None:
                        # End the current gap before this non-trading day
                        gap_end = current - timedelta(days=1)
                        gaps.append((gap_start, gap_end))
                        gap_start = None
                    non_trading_days_in_range.append(current)
            else:
                # Date exists in cache
                if gap_start is not None:
                    gap_end = current - timedelta(days=1)
                    gaps.append((gap_start, gap_end))
                    gap_start = None
            current += timedelta(days=1)
        
        # Check if gap extends to end
        if gap_start is not None:
            gaps.append((gap_start, max_date))
        
        # Filter gaps to only include trading days
        # Re-check gaps to ensure they only contain trading days
        filtered_gaps = []
        for gap_start, gap_end in gaps:
            trading_days_in_gap = []
            current = gap_start
            while current <= gap_end:
                try:
                    if is_trading_day(current):
                        trading_days_in_gap.append(current)
                except Exception:
                    # If check fails, include it
                    trading_days_in_gap.append(current)
                current += timedelta(days=1)
            
            if trading_days_in_gap:
                # Only add gap if it contains trading days
                filtered_gaps.append((gap_start, gap_end, len(trading_days_in_gap)))
        
        gaps = filtered_gaps
        
        # Count trading days in range for better coverage calculation
        trading_days_in_range = 0
        current = min_date
        while current <= max_date:
            try:
                if is_trading_day(current):
                    trading_days_in_range += 1
            except Exception:
                # If check fails, assume it's a trading day
                trading_days_in_range += 1
            current += timedelta(days=1)
        
        calendar_days_span = (max_date - min_date).days + 1
        trading_days_coverage = (count / trading_days_in_range * 100) if trading_days_in_range > 0 else 0
        calendar_days_coverage = (count / calendar_days_span * 100) if calendar_days_span > 0 else 0
        
        print(f"[OK] Cache file found: {cache_path}")
        print()
        print("CACHE STATISTICS:")
        print(f"  Total records: {count}")
        print(f"  Date range: {min_date} to {max_date}")
        print(f"  Calendar days span: {calendar_days_span} days")
        print(f"  Trading days in range: {trading_days_in_range} days")
        print(f"  Coverage (calendar days): {calendar_days_coverage:.1f}%")
        print(f"  Coverage (trading days): {trading_days_coverage:.1f}%")
        print()
        
        # Check how many have classifications
        if 'classification' in df.columns:
            has_classification = df['classification'].notna() & (df['classification'] != '')
            classification_count = has_classification.sum()
            print(f"  Records with classification: {classification_count} ({classification_count/count*100:.1f}%)")
        print()
        
        if gaps:
            total_trading_days_missing = sum(count for _, _, count in gaps)
            print(f"GAPS IN CACHE ({len(gaps)} gap(s), {total_trading_days_missing} trading days missing):")
            for gap_start, gap_end, trading_days_count in gaps[:20]:  # Show first 20 gaps
                gap_days = (gap_end - gap_start).days + 1
                print(f"  {gap_start} to {gap_end} ({gap_days} calendar days, {trading_days_count} trading days)")
            if len(gaps) > 20:
                print(f"  ... and {len(gaps) - 20} more gaps")
            print()
        else:
            print("GAPS: None (all trading days have data)")
            print()
        
        # Show non-trading days info
        if non_trading_days_in_range:
            print(f"NON-TRADING DAYS IN RANGE: {len(non_trading_days_in_range)} (not considered gaps)")
            print()
        
        # Show sample of dates
        print("SAMPLE DATES (first 10 and last 10):")
        for d in dates_sorted[:10]:
            if 'date' in df.columns:
                row = df[df['date'] == d].iloc[0]
                value = float(row['value'])
                classification = row.get('classification', '')
            else:
                value = float(df.loc[d, 'value'])
                classification = df.loc[d].get('classification', '')
            class_str = f" ({classification})" if classification and str(classification) != '' else ""
            print(f"  {d}: {value:.2f}{class_str}")
        if len(dates_sorted) > 20:
            print("  ...")
        for d in dates_sorted[-10:]:
            if 'date' in df.columns:
                row = df[df['date'] == d].iloc[0]
                value = float(row['value'])
                classification = row.get('classification', '')
            else:
                value = float(df.loc[d, 'value'])
                classification = df.loc[d].get('classification', '')
            class_str = f" ({classification})" if classification and str(classification) != '' else ""
            print(f"  {d}: {value:.2f}{class_str}")
        print()
        
        print("=" * 100)
        
    except Exception as e:
        print(f"[ERROR] Failed to read cache file: {e}")
        print(f"   Cache path: {cache_path}")


def wipe_cache() -> None:
    """Wipe the FGI cache (both file and in-memory)."""
    if not _PANDAS_AVAILABLE:
        print("[ERROR] pandas not available, cannot wipe cache")
        return
    
    # Initialize cache if needed
    initialize_cache()
    
    cache_path = _get_cache_path()
    if cache_path and cache_path.exists():
        try:
            cache_path.unlink()
            print(f"[OK] Wiped cache file: {cache_path}")
        except Exception as e:
            print(f"[ERROR] Failed to wipe cache file: {e}")
    else:
        print("[INFO] No cache file to wipe")
    
    # Clear in-memory cache
    import core.data.fear_greed_index as fgi_module
    fgi_module._in_memory_cache = {}
    print("[OK] Cleared in-memory cache")


def rebuild_cache_from_sources(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    only_trading_days: bool = True,
    overwrite_different: bool = True,
) -> Dict[str, int]:
    """
    Rebuild cache from CSV and CNN API.
    
    Order:
    1. Use CSV data first
    2. Use CNN API for missing dates or to update if value differs
    
    Args:
        start_date: Start date (default: from CSV min date)
        end_date: End date (default: today)
        only_trading_days: If True, only process trading days
        overwrite_different: If True, overwrite cache if value differs
    
    Returns:
        Dictionary with statistics: {"csv_added": count, "api_added": count, "updated": count}
    """
    if not _PANDAS_AVAILABLE:
        print("[ERROR] pandas not available, cannot rebuild cache")
        return {}
    
    # Initialize cache
    initialize_cache()
    
    print("=" * 100)
    print("REBUILDING CACHE FROM DATA SOURCES")
    print("=" * 100)
    print()
    
    # Load CSV data
    print("[INFO] Loading CSV data...")
    csv_data = load_fallback_csv()
    if not csv_data:
        print("[ERROR] Failed to load CSV data")
        return {}
    
    csv_dates = sorted(csv_data.keys())
    csv_min_date = csv_dates[0] if csv_dates else None
    csv_max_date = csv_dates[-1] if csv_dates else None
    
    print(f"[OK] Loaded {len(csv_data)} records from CSV")
    print(f"   Date range: {csv_min_date} to {csv_max_date}")
    print()
    
    # Determine date range
    if start_date is None:
        start_date = csv_min_date if csv_min_date else date(2011, 1, 1)
    if end_date is None:
        end_date = date.today()
    
    print(f"[INFO] Processing dates from {start_date} to {end_date}")
    if only_trading_days:
        print("[INFO] Only processing trading days")
    print()
    
    # Step 1: Add CSV data to cache
    print("=" * 100)
    print("STEP 1: Adding CSV data to cache")
    print("=" * 100)
    print()
    
    stats = {"csv_added": 0, "api_added": 0, "updated": 0, "csv_skipped": 0, "api_skipped": 0}
    
    current_date = start_date
    csv_processed = 0
    
    while current_date <= end_date:
        # Skip non-trading days if requested
        if only_trading_days:
            try:
                if not is_trading_day(current_date):
                    current_date += timedelta(days=1)
                    continue
            except Exception:
                pass
        
        # Check if date is in CSV
        if current_date in csv_data:
            csv_value, _ = csv_data[current_date]
            
            # Check cache
            cached_result = _load_from_cache(current_date)
            
            if cached_result is None:
                # Not in cache, add it
                _save_to_cache(current_date, csv_value, "")
                stats["csv_added"] += 1
                csv_processed += 1
                if csv_processed % 100 == 0:
                    print(f"   Progress: {csv_processed} CSV records added...")
            else:
                cached_value, _ = cached_result
                # Check if value differs
                if overwrite_different and abs(cached_value - csv_value) > 0.01:
                    _save_to_cache(current_date, csv_value, "")
                    stats["updated"] += 1
                    csv_processed += 1
                    if csv_processed % 100 == 0:
                        print(f"   Progress: {csv_processed} CSV records processed...")
                else:
                    stats["csv_skipped"] += 1
        
        current_date += timedelta(days=1)
    
    print()
    print(f"[OK] CSV processing complete: {stats['csv_added']} added, {stats['updated']} updated, {stats['csv_skipped']} skipped")
    print()
    
    # Step 2: Fetch from CNN API for all dates (to update cache if value differs)
    print("=" * 100)
    print("STEP 2: Fetching from CNN API")
    print("=" * 100)
    print()
    
    # Collect all dates to process
    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        if only_trading_days:
            try:
                if not is_trading_day(current_date):
                    current_date += timedelta(days=1)
                    continue
            except Exception:
                pass
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"[INFO] Will fetch {len(dates_to_process)} dates from CNN API")
    print(f"[INFO] For cache hits, will overwrite if value differs")
    print()
    
    api_processed = 0
    for test_date in dates_to_process:
        try:
            # Fetch from API
            api_result = fetch_cnn_api(test_date)
            
            if api_result and api_result[0] is not None:
                api_value, api_classification = api_result
                
                # Check what's currently in cache
                cached_result = _load_from_cache(test_date)
                
                if cached_result is None:
                    # Not in cache, add from API
                    _save_to_cache(test_date, api_value, api_classification or "")
                    stats["api_added"] += 1
                else:
                    cached_value, cached_classification = cached_result
                    # For cache hits, overwrite if value differs
                    if overwrite_different:
                        if abs(cached_value - api_value) > 0.01:
                            # Value differs, overwrite with API value
                            _save_to_cache(test_date, api_value, api_classification or "")
                            stats["updated"] += 1
                        else:
                            # Value matches, but update classification if available
                            if api_classification and not cached_classification:
                                _save_to_cache(test_date, api_value, api_classification)
                                stats["updated"] += 1
                            else:
                                stats["api_skipped"] += 1
                    else:
                        stats["api_skipped"] += 1
            else:
                stats["api_skipped"] += 1
            
            api_processed += 1
            if api_processed % 50 == 0:
                print(f"   Progress: {api_processed}/{len(dates_to_process)} API calls completed...")
        except Exception as e:
            print(f"   [WARN] API error for {test_date}: {e}")
            stats["api_skipped"] += 1
    
    print()
    print(f"[OK] API processing complete: {stats['api_added']} added, {stats['updated']} updated, {stats['api_skipped']} skipped")
    print()
    
    # Summary
    print("=" * 100)
    print("CACHE REBUILD SUMMARY")
    print("=" * 100)
    print()
    print(f"CSV records added: {stats['csv_added']}")
    print(f"CSV records skipped (already in cache): {stats['csv_skipped']}")
    print(f"API records added: {stats['api_added']}")
    print(f"Records updated: {stats['updated']}")
    print(f"API records skipped: {stats['api_skipped']}")
    print(f"Total records in cache: {stats['csv_added'] + stats['api_added'] + stats['updated']}")
    print()
    
    return stats


def get_data_source_ranges() -> Dict[str, Tuple[Optional[date], Optional[date], int]]:
    """
    Get date ranges for each data source.
    
    Returns:
        Dictionary mapping source name -> (min_date, max_date, count)
    """
    ranges = {}
    
    # Fallback CSV
    fallback_csv = load_fallback_csv()
    if fallback_csv:
        dates = sorted(fallback_csv.keys())
        ranges["Fallback CSV (fear-greed-2011-2023.csv)"] = (dates[0] if dates else None, dates[-1] if dates else None, len(dates))
    else:
        ranges["Fallback CSV (fear-greed-2011-2023.csv)"] = (None, None, 0)
    
    # CNN API - test a few dates to determine range
    # (We can't easily determine full range without testing many dates)
    ranges["CNN API"] = (None, None, None)  # Unknown without extensive testing
    
    # QuantConnect API
    ranges["QuantConnect API"] = (None, None, None)  # Unknown
    
    return ranges


def generate_sample_dates(
    start_date: date,
    end_date: date,
    num_dates: int,
    only_trading_days: bool = True,
) -> List[date]:
    """Generate random sample dates."""
    if only_trading_days:
        trading_days = []
        current = start_date
        while current <= end_date:
            try:
                if is_trading_day(current):
                    trading_days.append(current)
            except Exception:
                trading_days.append(current)
            current += timedelta(days=1)
        
        if len(trading_days) < num_dates:
            return random.sample(trading_days, len(trading_days))
        return random.sample(trading_days, num_dates)
    else:
        total_days = (end_date - start_date).days + 1
        return [start_date + timedelta(days=random.randint(0, total_days - 1)) for _ in range(num_dates)]


def test_all_sources(
    num_sample_dates: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    only_2025: bool = False,
    only_trading_days: bool = True,
    quantconnect_api_key: Optional[str] = None,
    rebuild_cache: bool = False,
) -> None:
    """
    Test all FGI data sources and compare results.
    
    Args:
        num_sample_dates: Number of sample dates to test
        start_date: Start date (default: 2011-01-01)
        end_date: End date (default: today)
        only_2025: If True, only test 2025 dates
        only_trading_days: If True, only test trading days
        quantconnect_api_key: Optional QuantConnect API key
    """
    print("=" * 100)
    print("FEAR & GREED INDEX - ALL DATA SOURCES COMPARISON")
    print("=" * 100)
    print()
    
    # Rebuild cache if requested
    if rebuild_cache:
        print("[INFO] Rebuild cache requested - wiping and rebuilding cache...")
        print()
        wipe_cache()
        print()
        
        # Set default dates for cache rebuild
        rebuild_start = start_date if start_date else date(2011, 1, 1)
        rebuild_end = end_date if end_date else date.today()
        
        if only_2025:
            rebuild_start = max(rebuild_start, date(2025, 1, 1))
            rebuild_end = min(rebuild_end, date(2025, 12, 31))
        
        rebuild_cache_from_sources(
            start_date=rebuild_start,
            end_date=rebuild_end,
            only_trading_days=only_trading_days,
            overwrite_different=True,
        )
        print()
        print("=" * 100)
        print()
        print("[INFO] Cache rebuild complete. Exiting (skipping random sampling).")
        return
    
    # Set default dates
    if start_date is None:
        start_date = date(2011, 1, 1)
    if end_date is None:
        end_date = date.today()
    
    if only_2025:
        start_date = max(start_date, date(2025, 1, 1))
        end_date = min(end_date, date(2025, 12, 31))
    
    print(f"[INFO] Testing date range: {start_date} to {end_date}")
    print(f"[INFO] Sample size: {num_sample_dates} dates")
    print(f"[INFO] Only trading days: {only_trading_days}")
    print(f"[INFO] Only 2025: {only_2025}")
    print()
    
    # Step 1: Show data source availability
    print("=" * 100)
    print("DATA SOURCE AVAILABILITY")
    print("=" * 100)
    print()
    
    ranges = get_data_source_ranges()
    print(f"{'Data Source':<40} {'Min Date':<12} {'Max Date':<12} {'Count':<10}")
    print("-" * 100)
    
    for source, (min_date, max_date, count) in ranges.items():
        min_str = str(min_date) if min_date else "N/A"
        max_str = str(max_date) if max_date else "N/A"
        count_str = str(count) if count is not None else "Unknown"
        print(f"{source:<40} {min_str:<12} {max_str:<12} {count_str:<10}")
    
    print()
    
    # Step 2: Generate sample dates
    print("=" * 100)
    print("GENERATING SAMPLE DATES")
    print("=" * 100)
    print()
    
    sample_dates = generate_sample_dates(start_date, end_date, num_sample_dates, only_trading_days)
    sample_dates.sort()
    print(f"[OK] Generated {len(sample_dates)} sample dates")
    print()
    
    # Step 3: Fetch data from all sources
    print("=" * 100)
    print("FETCHING DATA FROM ALL SOURCES")
    print("=" * 100)
    print()
    
    results = []
    source_stats = {
        "Fallback CSV": {"success": 0, "failed": 0},
        "CNN API": {"success": 0, "failed": 0},
        "QuantConnect API": {"success": 0, "failed": 0},
    }
    
    # Load CSV data once
    fallback_csv_data = load_fallback_csv()
    
    for i, test_date in enumerate(sample_dates, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(sample_dates)} dates...")
        
        result = {"date": test_date}
        
        # Fallback CSV
        if fallback_csv_data and test_date in fallback_csv_data:
            value, classification = fallback_csv_data[test_date]
            result["fallback_csv"] = (value, classification)
            source_stats["Fallback CSV"]["success"] += 1
        else:
            result["fallback_csv"] = None
            source_stats["Fallback CSV"]["failed"] += 1
        
        # CNN API
        try:
            cnn_result = fetch_cnn_api(test_date)
            if cnn_result and cnn_result[0] is not None:
                result["cnn_api"] = cnn_result
                source_stats["CNN API"]["success"] += 1
            else:
                result["cnn_api"] = None
                source_stats["CNN API"]["failed"] += 1
        except Exception:
            result["cnn_api"] = None
            source_stats["CNN API"]["failed"] += 1
        
        # QuantConnect API
        try:
            qc_result = fetch_quantconnect_api(test_date, quantconnect_api_key)
            if qc_result and qc_result[0] is not None:
                result["quantconnect_api"] = qc_result
                source_stats["QuantConnect API"]["success"] += 1
            else:
                result["quantconnect_api"] = None
                source_stats["QuantConnect API"]["failed"] += 1
        except Exception:
            result["quantconnect_api"] = None
            source_stats["QuantConnect API"]["failed"] += 1
        
        results.append(result)
    
    print()
    
    # Step 4: Print statistics
    print("=" * 100)
    print("SOURCE STATISTICS")
    print("=" * 100)
    print()
    print(f"{'Source':<25} {'Success':<10} {'Failed':<10} {'Success Rate':<15}")
    print("-" * 100)
    
    for source, stats in source_stats.items():
        total = stats["success"] + stats["failed"]
        rate = (stats["success"] / total * 100) if total > 0 else 0
        print(f"{source:<25} {stats['success']:<10} {stats['failed']:<10} {rate:.1f}%")
    
    print()
    
    # Step 5: Print comparison table
    print("=" * 100)
    print("DETAILED COMPARISON TABLE")
    print("=" * 100)
    print()
    
    # Find all unique values for each date
    matches = []
    mismatches = []
    csv_only = []
    api_only = []
    all_failed = []
    
    for result in results:
        test_date = result["date"]
        values = {}
        
        if result["fallback_csv"]:
            values["Fallback CSV"] = result["fallback_csv"][0]
        if result["cnn_api"]:
            values["CNN API"] = result["cnn_api"][0]
        if result["quantconnect_api"]:
            values["QuantConnect API"] = result["quantconnect_api"][0]
        
        if not values:
            all_failed.append(test_date)
        elif len(values) == 1:
            # Only one source has data
            source_name = list(values.keys())[0]
            if "CSV" in source_name:
                csv_only.append((test_date, source_name, list(values.values())[0]))
            else:
                api_only.append((test_date, source_name, list(values.values())[0]))
        else:
            # Multiple sources - check for matches
            unique_values = set(values.values())
            if len(unique_values) == 1:
                # All match
                matches.append((test_date, list(values.values())[0], list(values.keys())))
            else:
                # Mismatch
                mismatches.append((test_date, values))
    
    # Print table header
    print(f"{'Date':<12} {'Fallback CSV':<15} {'CNN API':<15} {'QC API':<15} {'Status':<20}")
    print("-" * 100)
    
    for result in results[:100]:  # Limit to first 100 for readability
        test_date = result["date"]
        
        fallback_str = f"{result['fallback_csv'][0]:.2f}" if result["fallback_csv"] else "N/A"
        cnn_str = f"{result['cnn_api'][0]:.2f}" if result["cnn_api"] else "N/A"
        qc_str = f"{result['quantconnect_api'][0]:.2f}" if result["quantconnect_api"] else "N/A"
        
        # Determine status
        values = []
        if result["fallback_csv"]:
            values.append(("Fallback CSV", result["fallback_csv"][0]))
        if result["cnn_api"]:
            values.append(("CNN API", result["cnn_api"][0]))
        if result["quantconnect_api"]:
            values.append(("QuantConnect API", result["quantconnect_api"][0]))
        
        if not values:
            status = "All failed"
        elif len(values) == 1:
            status = f"{values[0][0]} only"
        else:
            unique_vals = set(v[1] for v in values)
            if len(unique_vals) == 1:
                status = f"Match ({len(values)} sources)"
            else:
                status = f"Mismatch ({len(values)} sources)"
        
        print(f"{test_date} {fallback_str:<15} {cnn_str:<15} {qc_str:<15} {status:<20}")
    
    if len(results) > 100:
        print(f"... and {len(results) - 100} more dates")
    
    print()
    
    # Step 6: Print summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Total dates tested: {len(results)}")
    print(f"Matches (all sources agree): {len(matches)}")
    print(f"Mismatches (sources disagree): {len(mismatches)}")
    print(f"CSV only (not in APIs): {len(csv_only)}")
    print(f"API only (not in CSVs): {len(api_only)}")
    print(f"All failed: {len(all_failed)}")
    print()
    
    # Print mismatch details
    if mismatches:
        print("=" * 100)
        print("MISMATCHES (First 20)")
        print("=" * 100)
        print()
        for test_date, values in mismatches[:20]:
            print(f"Date: {test_date}")
            for source, val in values.items():
                print(f"  {source}: {val:.2f}")
            print()
    
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test all Fear & Greed Index data sources"
    )
    parser.add_argument(
        "--num-dates",
        type=int,
        default=50,
        help="Number of sample dates to test (default: 50)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD, default: 2011-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--only-2025",
        action="store_true",
        help="Only test 2025 dates"
    )
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        help="Include weekends and holidays (default: only trading days)"
    )
    parser.add_argument(
        "--qc-api-key",
        type=str,
        help="QuantConnect API key (optional)"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Wipe and rebuild cache from CSV and CNN API (order: CSV first, then API)"
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Check and display cache date range and statistics"
    )
    parser.add_argument(
        "--fill-cache",
        type=str,
        help="Fill cache for a specific date range (format: START:END, e.g., 2011-01-08:2011-01-09)"
    )
    
    args = parser.parse_args()
    
    # If only checking cache, do that and exit
    if args.check_cache and not args.rebuild_cache and not args.fill_cache:
        check_cache_range()
        sys.exit(0)
    
    # Handle --fill-cache parameter
    if args.fill_cache:
        try:
            # Parse date range (format: START:END)
            parts = args.fill_cache.split(":")
            if len(parts) != 2:
                print("[ERROR] --fill-cache format must be START:END (e.g., 2011-01-08:2011-01-09)")
                sys.exit(1)
            
            fill_start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").date()
            fill_end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").date()
            
            if fill_start > fill_end:
                print("[ERROR] Start date must be before or equal to end date")
                sys.exit(1)
            
            print("=" * 100)
            print("FILLING CACHE FOR DATE RANGE")
            print("=" * 100)
            print()
            print(f"Date range: {fill_start} to {fill_end}")
            print()
            
            # Fill cache for the specified range
            stats = rebuild_cache_from_sources(
                start_date=fill_start,
                end_date=fill_end,
                only_trading_days=not args.include_weekends,
                overwrite_different=True,
            )
            
            print()
            print("=" * 100)
            print("CACHE FILL COMPLETE")
            print("=" * 100)
            print()
            print(f"CSV records added: {stats['csv_added']}")
            print(f"CSV records updated: {stats['updated']}")
            print(f"CSV records skipped: {stats['csv_skipped']}")
            print(f"API records added: {stats['api_added']}")
            print(f"API records skipped: {stats['api_skipped']}")
            print(f"Total records processed: {stats['csv_added'] + stats['api_added'] + stats['updated']}")
            print()
            
            sys.exit(0)
        except ValueError as e:
            print(f"[ERROR] Invalid date format in --fill-cache: {e}")
            print("Expected format: YYYY-MM-DD:YYYY-MM-DD (e.g., 2011-01-08:2011-01-09)")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to fill cache: {e}")
            sys.exit(1)
    
    start_date = None
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    
    end_date = None
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    test_all_sources(
        num_sample_dates=args.num_dates,
        start_date=start_date,
        end_date=end_date,
        only_2025=args.only_2025,
        only_trading_days=not args.include_weekends,
        quantconnect_api_key=args.qc_api_key,
        rebuild_cache=args.rebuild_cache,
    )

