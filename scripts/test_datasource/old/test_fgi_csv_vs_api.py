"""
Test script to compare fear_greed_index.csv against CNN API and update with missing 2025 data.

This script:
1. Identifies mismatches between CSV and API values
2. Identifies gaps in values (dates that exist in one but not the other)
3. Fetches missing 2025 data from API and appends to CSV
4. Prints detailed comparison results
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Tuple, List, Optional
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data.fear_greed_index import _get_historical_fgi_from_api, _load_csv_data
from core.data import is_trading_day


def normalize_classification(classification: Optional[str]) -> str:
    """Normalize classification string for comparison."""
    if classification is None:
        return ""
    # Convert to lowercase and strip whitespace
    normalized = str(classification).lower().strip()
    # Handle variations
    if normalized in ["extreme fear", "extremefear"]:
        return "extreme fear"
    elif normalized in ["extreme greed", "extremegreed"]:
        return "extreme greed"
    return normalized


def date_to_timestamp(target_date: date) -> int:
    """Convert date to Unix timestamp (start of day in UTC)."""
    dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def format_classification(classification: str) -> str:
    """Format classification to match CSV format (title case)."""
    if not classification:
        return ""
    # Convert to title case
    formatted = classification.title()
    # Handle special cases
    if formatted.lower() in ["extreme fear", "extremefear"]:
        return "Extreme Fear"
    elif formatted.lower() in ["extreme greed", "extremegreed"]:
        return "Extreme Greed"
    elif formatted.lower() in ["fear"]:
        return "Fear"
    elif formatted.lower() in ["greed"]:
        return "Greed"
    elif formatted.lower() in ["neutral"]:
        return "Neutral"
    return formatted


def fetch_and_append_missing_data(
    csv_file_path: Path,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
    only_2025: bool = True,
    skip_non_trading: bool = True,
) -> List[Tuple[date, float, str]]:
    """
    Fetch missing data from API and append to CSV.
    
    Args:
        csv_file_path: Path to CSV file
        start_date: Start date to fetch from
        end_date: End date to fetch to
        dry_run: If True, don't write to file, just return data
        only_2025: If True, only process 2025 dates
        skip_non_trading: If True, skip non-trading days
    
    Returns:
        List of tuples (date, value, classification) that were fetched
    """
    print("=" * 80)
    print("FETCHING MISSING DATA FROM API")
    print("=" * 80)
    print()
    
    # Load existing CSV data
    csv_data = _load_csv_data()
    if csv_data is None:
        print("[ERROR] Failed to load CSV data")
        return []
    
    csv_date_set = set(csv_data.keys())
    
    # Find missing dates (only 2025, only trading days)
    missing_dates = []
    current_date = start_date
    while current_date <= end_date:
        # Only process 2025 dates
        if only_2025 and current_date.year != 2025:
            current_date += timedelta(days=1)
            continue
        
        # Skip non-trading days
        if skip_non_trading:
            try:
                if not is_trading_day(current_date):
                    current_date += timedelta(days=1)
                    continue
            except Exception:
                # If trading day check fails, include the date
                pass
        
        if current_date not in csv_date_set:
            missing_dates.append(current_date)
        current_date += timedelta(days=1)
    
    if not missing_dates:
        print(f"[INFO] No missing dates found between {start_date} and {end_date}")
        return []
    
    print(f"[INFO] Found {len(missing_dates)} missing dates")
    print(f"[INFO] Fetching data from API...")
    print()
    
    fetched_data = []
    errors = []
    
    for i, target_date in enumerate(missing_dates, 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(missing_dates)} dates fetched...")
        
        try:
            api_value, api_classification = _get_historical_fgi_from_api(target_date)
            
            if api_value is not None:
                classification = format_classification(api_classification or "")
                fetched_data.append((target_date, api_value, classification))
            else:
                errors.append((target_date, "No data from API"))
        except Exception as e:
            errors.append((target_date, str(e)))
    
    print()
    print(f"[OK] Successfully fetched {len(fetched_data)} dates")
    if errors:
        print(f"[WARN] Failed to fetch {len(errors)} dates")
        for err_date, err_msg in errors[:10]:
            print(f"   {err_date}: {err_msg}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    print()
    
    if not fetched_data:
        print("[INFO] No data to append")
        return []
    
    if dry_run:
        print("[DRY RUN] Would append the following data:")
        for d, v, c in fetched_data[:10]:
            print(f"   {d}: {v:.2f} ({c})")
        if len(fetched_data) > 10:
            print(f"   ... and {len(fetched_data) - 10} more")
        return fetched_data
    
    # Append to CSV
    print("[INFO] Appending data to CSV...")
    
    try:
        # Read existing CSV
        df_existing = pd.read_csv(csv_file_path)
        
        # Create new rows
        new_rows = []
        for target_date, value, classification in fetched_data:
            timestamp = date_to_timestamp(target_date)
            date_str = target_date.strftime("%Y-%m-%d")
            new_rows.append({
                "timestamp": timestamp,
                "value": value,
                "classification": classification,
                "date": date_str,
            })
        
        # Create DataFrame for new data
        df_new = pd.DataFrame(new_rows)
        
        # Combine and sort by timestamp
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.sort_values("timestamp")
        
        # Remove duplicates (keep first occurrence)
        df_combined = df_combined.drop_duplicates(subset=["date"], keep="first")
        
        # Write back to CSV
        df_combined.to_csv(csv_file_path, index=False)
        
        print(f"[OK] Successfully appended {len(fetched_data)} records to CSV")
        print(f"[INFO] CSV now contains {len(df_combined)} total records")
        
        return fetched_data
        
    except Exception as e:
        print(f"[ERROR] Failed to append data to CSV: {e}")
        return []


def compare_csv_vs_api(
    csv_file_path: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    max_api_calls: int = 1000,
) -> None:
    """
    Compare CSV data against API data.
    
    Args:
        csv_file_path: Path to CSV file. If None, uses default location.
        start_date: Start date for comparison. If None, uses CSV min date.
        end_date: End date for comparison. If None, uses CSV max date.
        max_api_calls: Maximum number of API calls to make (to avoid rate limiting).
    """
    print("=" * 80)
    print("FGI CSV vs API Comparison")
    print("=" * 80)
    print()
    
    # Load CSV data
    if csv_file_path is None:
        csv_file_path = project_root / "core" / "data" / "fear_greed_index.csv"
    
    if not csv_file_path.exists():
        print(f"[ERROR] CSV file not found: {csv_file_path}")
        return
    
    print(f"[INFO] Loading CSV from: {csv_file_path}")
    csv_data = _load_csv_data()
    
    if csv_data is None or not csv_data:
        print("[ERROR] Failed to load CSV data or CSV is empty")
        return
    
    csv_dates = sorted(csv_data.keys())
    csv_min_date = csv_dates[0]
    csv_max_date = csv_dates[-1]
    
    print(f"[OK] Loaded {len(csv_data)} records from CSV")
    print(f"   Date range: {csv_min_date} to {csv_max_date}")
    print()
    
    # Determine comparison date range (default to 2025 only)
    if start_date is None:
        # Start from 2025-01-01 if CSV has 2025 data, otherwise use CSV min
        if csv_max_date.year >= 2025:
            start_date = date(2025, 1, 1)
        else:
            start_date = csv_min_date
    if end_date is None:
        # End at 2025-12-31 if CSV has 2025 data, otherwise use CSV max
        if csv_max_date.year >= 2025:
            end_date = date(2025, 12, 31)
        else:
            end_date = csv_max_date
    
    print(f"[INFO] Comparing dates from {start_date} to {end_date}")
    print(f"   (Limited to {max_api_calls} API calls)")
    print()
    
    # Filter CSV dates to comparison range (only 2025, only trading days)
    comparison_dates = []
    for d in csv_dates:
        if start_date <= d <= end_date:
            # Only process 2025 dates
            if d.year != 2025:
                continue
            # Skip non-trading days
            try:
                if not is_trading_day(d):
                    continue
            except Exception:
                # If trading day check fails, include the date
                pass
            comparison_dates.append(d)
    
    if len(comparison_dates) > max_api_calls:
        print(f"[WARN] Limiting comparison to first {max_api_calls} dates (out of {len(comparison_dates)})")
        comparison_dates = comparison_dates[:max_api_calls]
    
    print(f"[INFO] Comparing {len(comparison_dates)} dates (2025 trading days only)...")
    print()
    
    # Track results
    matches: List[Tuple[date, float, str]] = []
    value_mismatches: List[Tuple[date, float, float, str, str]] = []
    classification_mismatches: List[Tuple[date, float, str, str]] = []
    csv_only: List[Tuple[date, float, str]] = []
    api_errors: List[Tuple[date, str]] = []
    
    # Compare each date
    for i, target_date in enumerate(comparison_dates, 1):
        if i % 50 == 0:
            print(f"   Progress: {i}/{len(comparison_dates)} dates checked...")
        
        csv_value, csv_classification = csv_data[target_date]
        csv_class_norm = normalize_classification(csv_classification)
        
        # Fetch from API
        try:
            api_value, api_classification = _get_historical_fgi_from_api(target_date)
            api_class_norm = normalize_classification(api_classification)
            
            if api_value is None:
                # Date exists in CSV but not in API
                csv_only.append((target_date, csv_value, csv_classification or ""))
            else:
                # Both exist - compare values
                value_diff = abs(csv_value - api_value)
                value_match = value_diff < 0.01  # Allow small floating point differences
                class_match = csv_class_norm == api_class_norm
                
                if value_match and class_match:
                    matches.append((target_date, csv_value, csv_classification or ""))
                else:
                    if not value_match:
                        value_mismatches.append((
                            target_date,
                            csv_value,
                            api_value,
                            csv_classification or "",
                            api_classification or "",
                        ))
                    if not class_match:
                        classification_mismatches.append((
                            target_date,
                            csv_value,
                            csv_classification or "",
                            api_classification or "",
                        ))
        except Exception as e:
            api_errors.append((target_date, str(e)))
    
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Print summary
    print("[SUMMARY]")
    print("-" * 80)
    print(f"Total dates compared: {len(comparison_dates)}")
    print(f"[OK] Matches: {len(matches)}")
    print(f"[ERROR] Value mismatches: {len(value_mismatches)}")
    print(f"[WARN] Classification mismatches: {len(classification_mismatches)}")
    print(f"[INFO] CSV only (not in API): {len(csv_only)}")
    print(f"[ERROR] API errors: {len(api_errors)}")
    print()
    
    # Print value mismatches
    if value_mismatches:
        print("=" * 80)
        print("VALUE MISMATCHES")
        print("=" * 80)
        print(f"{'Date':<12} {'CSV Value':<12} {'API Value':<12} {'Diff':<12} {'CSV Class':<20} {'API Class':<20}")
        print("-" * 80)
        for target_date, csv_val, api_val, csv_class, api_class in value_mismatches[:50]:  # Limit to first 50
            diff = csv_val - api_val
            print(f"{target_date} {csv_val:<12.2f} {api_val:<12.2f} {diff:<12.2f} {csv_class:<20} {api_class:<20}")
        if len(value_mismatches) > 50:
            print(f"... and {len(value_mismatches) - 50} more value mismatches")
        print()
    
    # Print classification mismatches (excluding those already in value_mismatches)
    classification_only = [
        (d, v, c1, c2) for d, v, c1, c2 in classification_mismatches
        if not any(d == dm[0] for dm in value_mismatches)
    ]
    if classification_only:
        print("=" * 80)
        print("CLASSIFICATION MISMATCHES (same value, different classification)")
        print("=" * 80)
        print(f"{'Date':<12} {'Value':<12} {'CSV Class':<20} {'API Class':<20}")
        print("-" * 80)
        for target_date, val, csv_class, api_class in classification_only[:50]:  # Limit to first 50
            print(f"{target_date} {val:<12.2f} {csv_class:<20} {api_class:<20}")
        if len(classification_only) > 50:
            print(f"... and {len(classification_only) - 50} more classification mismatches")
        print()
    
    # Print CSV-only dates
    if csv_only:
        print("=" * 80)
        print("DATES IN CSV BUT NOT IN API")
        print("=" * 80)
        print(f"{'Date':<12} {'Value':<12} {'Classification':<20}")
        print("-" * 80)
        for target_date, val, classification in csv_only[:50]:  # Limit to first 50
            print(f"{target_date} {val:<12.2f} {classification:<20}")
        if len(csv_only) > 50:
            print(f"... and {len(csv_only) - 50} more dates")
        print()
    
    # Print API errors
    if api_errors:
        print("=" * 80)
        print("API ERRORS")
        print("=" * 80)
        print(f"{'Date':<12} {'Error':<60}")
        print("-" * 80)
        for target_date, error in api_errors[:50]:  # Limit to first 50
            error_short = error[:60] if len(error) > 60 else error
            print(f"{target_date} {error_short:<60}")
        if len(api_errors) > 50:
            print(f"... and {len(api_errors) - 50} more errors")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare FGI CSV against CNN API and update missing 2025 data")
    parser.add_argument("--csv", type=str, help="Path to CSV file (default: core/data/fear_greed_index.csv)")
    parser.add_argument("--start", type=str, help="Start date for comparison/fetch (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for comparison/fetch (YYYY-MM-DD)")
    parser.add_argument("--max-calls", type=int, default=1000, help="Maximum API calls for comparison (default: 1000)")
    parser.add_argument("--update", action="store_true", help="Fetch and append missing 2025 data to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't write to CSV)")
    parser.add_argument("--compare-only", action="store_true", help="Only run comparison, don't update")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv) if args.csv else (project_root / "core" / "data" / "fear_greed_index.csv")
    
    # Load CSV to find last date
    csv_data = _load_csv_data()
    if csv_data:
        csv_dates = sorted(csv_data.keys())
        csv_max_date = csv_dates[-1]
        today = date.today()
        
        print(f"[INFO] CSV last date: {csv_max_date}")
        print(f"[INFO] Today's date: {today}")
        print()
        
        # If update mode, fetch missing 2025 data
        if args.update and not args.compare_only:
            # Determine date range for fetching
            if args.start:
                fetch_start = datetime.strptime(args.start, "%Y-%m-%d").date()
            else:
                # Start from day after last CSV date, but only for 2025
                if csv_max_date.year >= 2025:
                    fetch_start = csv_max_date + timedelta(days=1)
                else:
                    fetch_start = date(2025, 1, 1)
                
                # Ensure we don't go beyond 2025
                if fetch_start.year > 2025:
                    fetch_start = date(2025, 12, 31) + timedelta(days=1)  # Will be skipped
                elif fetch_start < date(2025, 1, 1):
                    fetch_start = date(2025, 1, 1)
            
            if args.end:
                fetch_end = datetime.strptime(args.end, "%Y-%m-%d").date()
            else:
                fetch_end = min(today, date(2025, 12, 31))  # Don't go beyond 2025
            
            # Only fetch if there are dates to fetch and we're in 2025
            if fetch_start <= fetch_end and fetch_start.year == 2025:
                print(f"[INFO] Fetching missing data from {fetch_start} to {fetch_end}")
                print()
                
                fetched = fetch_and_append_missing_data(
                    csv_file_path=csv_path,
                    start_date=fetch_start,
                    end_date=fetch_end,
                    dry_run=args.dry_run,
                    only_2025=True,
                    skip_non_trading=True,
                )
                
                if fetched and not args.dry_run:
                    print()
                    print("[INFO] Reloading CSV to verify update...")
                    # Reload CSV data
                    from core.data.fear_greed_index import _csv_data_cache, _csv_file_path
                    # Clear cache to force reload
                    import core.data.fear_greed_index as fgi_module
                    fgi_module._csv_data_cache = None
                    fgi_module._csv_file_path = None
            else:
                print(f"[INFO] No dates to fetch (fetch_start {fetch_start} > fetch_end {fetch_end})")
        
        # Run comparison if requested or if not in update-only mode
        if not args.update or args.compare_only:
            start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
            end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
            
            compare_csv_vs_api(
                csv_file_path=csv_path,
                start_date=start,
                end_date=end,
                max_api_calls=args.max_calls,
            )
    else:
        print("[ERROR] Failed to load CSV data")
