"""
Test script to fetch and display CNN Fear & Greed Index.

Tests:
- Current FGI from fear-and-greed Python package
- Historical FGI from CNN API (https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD)
- Historical FGI from local CSV file (core/data/fear_greed_index.csv) for comparison

Installation: pip install fear-and-greed httpx
Documentation: 
  - fear-and-greed: https://pypi.org/project/fear-and-greed/
  - CNN API: https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import httpx

# Add project root to Python path (if needed for any future imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import fear_and_greed
except ImportError:
    fear_and_greed = None


def format_index_description(description: str) -> str:
    """Format the index description for better readability."""
    # Capitalize first letter of each word
    return description.replace('_', ' ').title()


def get_fear_greed_color(value: float) -> str:
    """Return a color indicator based on the index value."""
    if value <= 25:
        return "[RED]"  # Extreme Fear
    elif value <= 45:
        return "[ORANGE]"  # Fear
    elif value <= 55:
        return "[YELLOW]"  # Neutral
    elif value <= 75:
        return "[GREEN]"  # Greed
    else:
        return "[GREEN++]"  # Extreme Greed


def fetch_historical_fgi_from_api(target_date: date, max_retries: int = 2) -> tuple[float | None, str | None, str]:
    """
    Fetch historical FGI data from CNN API for a specific date.
    
    API endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD
    
    Args:
        target_date: Date to fetch FGI for
        max_retries: Maximum number of retry attempts for transient errors
    
    Returns:
        Tuple of (value, classification, error_message)
        If successful, error_message will be empty string
    """
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date_str}"
    
    # Headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com",
    }
    
    import time
    
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                
                if response.status_code == 418:
                    return None, None, "API blocked request (418 - bot detection). Try again later."
                
                # Handle 500 errors with retry
                if response.status_code == 500:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        return None, None, f"API server error (500) - data may not be available for this date"
                
                # Handle 404 - data not available for this date
                if response.status_code == 404:
                    return None, None, f"Data not available (404) - date may be too old or invalid"
                
                response.raise_for_status()
            
            # Try to parse as JSON
            try:
                data = response.json()
            except Exception:
                # If not JSON, try to extract from text
                text = response.text
                return None, None, f"API returned non-JSON response: {text[:200]}"
            
            # Parse the response - API returns:
            # {
            #   "fear_and_greed": { "score": value, "rating": classification, ... },
            #   "fear_and_greed_historical": { "data": [ { "x": timestamp_ms, "y": fgi_value, "rating": classification }, ... ] }
            # }
            value = None
            classification = None
            
            # Convert target date to timestamp (start of day in UTC)
            from datetime import timezone
            target_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            target_timestamp_ms = int(target_datetime.timestamp() * 1000)
            
            if isinstance(data, dict):
                # First, try to get from fear_and_greed_historical.data array
                if "fear_and_greed_historical" in data:
                    historical = data["fear_and_greed_historical"]
                    if isinstance(historical, dict) and "data" in historical:
                        data_array = historical["data"]
                        if isinstance(data_array, list):
                            # Find the data point that matches our target date
                            # Look for exact match or closest match within 24 hours
                            best_match = None
                            min_diff = float('inf')
                            
                            for point in data_array:
                                if isinstance(point, dict) and "x" in point and "y" in point:
                                    point_timestamp = float(point["x"])
                                    diff = abs(point_timestamp - target_timestamp_ms)
                                    
                                    # If exact match or within 24 hours, use it
                                    if diff < min_diff and diff < 86400000:  # 24 hours in ms
                                        min_diff = diff
                                        best_match = point
                            
                            if best_match:
                                # The 'y' field IS the FGI value (0-100)
                                value = float(best_match["y"])
                                if "rating" in best_match:
                                    classification = str(best_match["rating"])
                
                # Fallback: if not found in historical data, try current fear_and_greed score
                # (though this won't be historical, it's a fallback)
                if value is None and "fear_and_greed" in data:
                    current = data["fear_and_greed"]
                    if isinstance(current, dict) and "score" in current:
                        value = float(current["score"])
                        if "rating" in current:
                            classification = str(current["rating"])
            
            if value is not None:
                return value, classification, ""
            else:
                # Return the structure for debugging
                structure_info = "Unknown structure"
                if isinstance(data, dict):
                    structure_info = f"Keys: {list(data.keys())[:10]}"
                elif isinstance(data, list):
                    structure_info = f"List with {len(data)} items"
                return None, None, f"Could not parse FGI value from API response. {structure_info}"
            
            # If we get here, the request was successful - exit retry loop
            break
                
        except httpx.HTTPStatusError as e:
            # Already handled 404 and 500 above, but catch other HTTP errors
            if e.response.status_code == 404:
                return None, None, f"Data not available (404) - date may be too old or invalid"
            elif e.response.status_code == 500:
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2
                    if attempt > 0:
                        print(f"(retry {attempt + 1}/{max_retries} after {wait_time}s)", end=" ", flush=True)
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    return None, None, f"API server error (500) - data may not be available for this date"
            else:
                return None, None, f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except httpx.TimeoutException:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                if attempt > 0:
                    print(f"(retry {attempt + 1}/{max_retries} after {wait_time}s)", end=" ", flush=True)
                time.sleep(wait_time)
                continue  # Retry
            else:
                return None, None, "Request timeout (after retries)"
        except Exception as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                if attempt > 0:
                    print(f"(retry {attempt + 1}/{max_retries} after {wait_time}s)", end=" ", flush=True)
                time.sleep(wait_time)
                continue  # Retry
    else:
                return None, None, f"Error fetching from API: {str(e)}"
    
    # Should not reach here, but return error if we do
    return None, None, "Failed to fetch data after retries"


def load_historical_fgi(csv_path: Path) -> pd.DataFrame:
    """Load historical FGI data from CSV file."""
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        # Parse date column
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # Ensure value is numeric
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Remove rows with invalid values
        df = df.dropna(subset=["value", "date"])
        # Sort by date
        df = df.sort_values("date")
        # Set date as index for faster lookups
        df = df.set_index("date")
        return df
    except Exception as e:
        print(f"Error loading historical FGI data: {e}")
        return pd.DataFrame()


def get_historical_fgi_value(df: pd.DataFrame, target_date: date, forward_fill: bool = True) -> tuple[float | None, str | None]:
    """
    Get historical FGI value for a specific date.
    
    Returns:
        Tuple of (value, classification) or (None, None) if not found
    """
    if df.empty:
        return None, None
    
    # Check date range
    min_date = df.index.min()
    max_date = df.index.max()
    
    if target_date < min_date:
        return None, None
    if target_date > max_date:
        if forward_fill:
            # Use most recent value
            return float(df.iloc[-1]["value"]), str(df.iloc[-1].get("classification", "N/A"))
        return None, None
    
    # Try exact match
    if target_date in df.index:
        row = df.loc[target_date]
        value = float(row["value"])
        classification = str(row.get("classification", "N/A")) if "classification" in df.columns else None
        return value, classification
    
    # Forward fill from previous dates
    if forward_fill:
        prev_dates = df.index[df.index <= target_date]
        if len(prev_dates) > 0:
            row = df.loc[prev_dates[-1]]
            value = float(row["value"])
            classification = str(row.get("classification", "N/A")) if "classification" in df.columns else None
            return value, classification
    
    return None, None


def test_current_fgi():
    """Test current FGI from fear-and-greed package."""
    if fear_and_greed is None:
        print("⚠️  fear-and-greed package not installed. Skipping current FGI test.")
        print("   Install with: pip install fear-and-greed")
        return False
    
    try:
        print("Fetching current CNN Fear & Greed Index...")
        index_data = fear_and_greed.get()
        
        print()
        print("=" * 60)
        print("Current Fear & Greed Index (from package)")
        print("=" * 60)
        print()
        
        # Display the index value with color indicator
        color = get_fear_greed_color(index_data.value)
        print(f"Index Value: {color} {index_data.value:.2f} / 100")
        print()
        
        # Display description
        formatted_desc = format_index_description(index_data.description)
        print(f"Category: {formatted_desc}")
        print()
        
        # Display last update timestamp
        if index_data.last_update:
            # Convert to local time if timezone-aware
            if index_data.last_update.tzinfo:
                local_time = index_data.last_update.astimezone()
            else:
                local_time = index_data.last_update
            
            print(f"Last Updated: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  (UTC: {index_data.last_update.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        else:
            print("Last Updated: N/A")
        
        print()
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("Error fetching current Fear & Greed Index")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Possible causes:")
        print("  - Network connectivity issues")
        print("  - CNN website temporarily unavailable")
        print("  - Rate limiting (requests are cached for 1 minute)")
        print()
        return False


def test_historical_fgi_api(test_date: date):
    """Test historical FGI from CNN API."""
    print()
    print("=" * 60)
    print(f"Historical Fear & Greed Index (from CNN API)")
    print(f"Date: {test_date}")
    print("=" * 60)
    print()
    
    print(f"Fetching FGI data for {test_date} from CNN API...")
    value, classification, error = fetch_historical_fgi_from_api(test_date)
    
    if value is not None:
        color = get_fear_greed_color(value)
        print(f"FGI Value: {color} {value:.2f} / 100")
        if classification and classification != "N/A":
            print(f"Classification: {classification}")
        print()
        return True
    else:
        print(f"[WARNING] Failed to fetch FGI data: {error}")
        print()
        return False


def test_historical_fgi_csv(csv_path: Path, test_date: date | None = None):
    """Test historical FGI from CSV file."""
    print()
    print("=" * 60)
    print("Historical Fear & Greed Index (from CSV)")
    print("=" * 60)
    print()
    
    # Load historical data
    df = load_historical_fgi(csv_path)
    
    if df.empty:
        print(f"⚠️  Historical FGI data not found at: {csv_path}")
        return False
    
    min_date = df.index.min()
    max_date = df.index.max()
    total_records = len(df)
    
    print(f"Data Range: {min_date} to {max_date}")
    print(f"Total Records: {total_records:,}")
    print()
    
    if test_date:
        # Test specific date
        print(f"Testing date: {test_date}")
        value, classification = get_historical_fgi_value(df, test_date)
        
        if value is not None:
            color = get_fear_greed_color(value)
            print(f"FGI Value: {color} {value:.2f} / 100")
            if classification and classification != "N/A":
                print(f"Classification: {classification}")
            print()
        else:
            print(f"⚠️  No FGI data found for {test_date}")
            print(f"   Available range: {min_date} to {max_date}")
            print()
    else:
        # Show sample of recent dates
        print("Sample of Recent Dates:")
        print("-" * 60)
        recent_dates = df.tail(10)
        for idx_date, row in recent_dates.iterrows():
            value = float(row["value"])
            color = get_fear_greed_color(value)
            classification = str(row.get("classification", "N/A")) if "classification" in df.columns else "N/A"
            print(f"{idx_date}: {color} {value:.2f} / 100 ({classification})")
        print()
        
        # Show some historical dates
        print("Sample of Historical Dates (from 2020):")
        print("-" * 60)
        historical_2020 = df[df.index >= date(2020, 1, 1)].head(10)
        for idx_date, row in historical_2020.iterrows():
            value = float(row["value"])
            color = get_fear_greed_color(value)
            classification = str(row.get("classification", "N/A")) if "classification" in df.columns else "N/A"
            print(f"{idx_date}: {color} {value:.2f} / 100 ({classification})")
        print()
    
    return True


def update_csv_with_missing_data(csv_path: Path, start_date: date | None = None, end_date: date | None = None) -> int:
    """
    Fetch missing FGI data from API and append to CSV file.
    
    Args:
        csv_path: Path to CSV file
        start_date: Start date for range to check (default: CSV min date or 2018-02-01)
        end_date: End date for range to check (default: today)
    
    Returns:
        Number of new records added
    """
    print()
    print("=" * 60)
    print("Updating CSV with Missing FGI Data")
    print("=" * 60)
    print()
    
    # Load existing CSV data
    df = load_historical_fgi(csv_path)
    
    # Determine date range to check
    if df.empty:
        print("⚠️  CSV file is empty or doesn't exist.")
        if start_date is None:
            start_date = date(2018, 2, 1)  # Default start date
        if end_date is None:
            end_date = date.today()
        print(f"Will fetch data from {start_date} to {end_date}")
        existing_dates = set()
    else:
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = date.today()
        existing_dates = set(df.index)
        print(f"Checking for missing data from {start_date} to {end_date}")
        print(f"Existing CSV data: {df.index.min()} to {df.index.max()} ({len(df)} records)")
    
    print()
    
    # Generate all dates in range (excluding weekends for market days)
    from datetime import timedelta
    missing_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:  # Monday=0 to Friday=4
            if current_date not in existing_dates:
                missing_dates.append(current_date)
        current_date += timedelta(days=1)
    
    if not missing_dates:
        print("✅ No missing dates found. CSV is up to date.")
        return 0
    
    print(f"Found {len(missing_dates)} missing dates")
    print(f"First 10 missing dates: {missing_dates[:10]}")
    if len(missing_dates) > 10:
        print(f"... and {len(missing_dates) - 10} more")
    print()
    
    # Fetch missing data from API
    new_records = []
    success_count = 0
    fail_count = 0
    
    print("Fetching missing data from CNN API...")
    print("-" * 60)
    
    for i, target_date in enumerate(missing_dates, 1):
        print(f"[{i}/{len(missing_dates)}] Fetching {target_date}...", end=" ", flush=True)
        
        value, classification, error = fetch_historical_fgi_from_api(target_date)
        
        if value is not None:
            # Calculate timestamp (start of day in UTC)
            from datetime import timezone
            target_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            timestamp = int(target_datetime.timestamp())
            
            new_records.append({
                "timestamp": timestamp,
                "value": value,
                "classification": classification or "",
                "date": target_date.strftime("%Y-%m-%d")
            })
            success_count += 1
            print(f"✅ {value:.2f} ({classification or 'N/A'})")
        else:
            fail_count += 1
            print(f"❌ {error}")
        
        # Add small delay to avoid rate limiting
        if i < len(missing_dates):
            import time
            time.sleep(0.5)  # 500ms delay between requests
    
    print()
    print("-" * 60)
    print(f"Fetched {success_count} records, {fail_count} failed")
    print()
    
    if not new_records:
        print("⚠️  No new records to add to CSV.")
        return 0
    
    # Append new records to CSV
    try:
        # Create DataFrame from new records
        new_df = pd.DataFrame(new_records)
        
        if new_df.empty:
            print("⚠️  No new records to add.")
            return 0
        
        # Load existing CSV (as DataFrame, not indexed) if it exists
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            # Combine and sort by date
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # Create new CSV with just the new records
            combined_df = new_df
        
        # Ensure date column is datetime for sorting
        combined_df["date"] = pd.to_datetime(combined_df["date"])
        combined_df = combined_df.sort_values("date")
        
        # Remove duplicates (keep first occurrence)
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
        
        # Convert date back to string format
        combined_df["date"] = combined_df["date"].dt.strftime("%Y-%m-%d")
        
        # Ensure columns are in correct order
        column_order = ["timestamp", "value", "classification", "date"]
        combined_df = combined_df[column_order]
        
        # Save to CSV
        combined_df.to_csv(csv_path, index=False)
        
        print(f"✅ Added {len(new_records)} new records to {csv_path}")
        print(f"   Total records in CSV: {len(combined_df)}")
        print()
        
        return len(new_records)
        
    except Exception as e:
        print(f"❌ Error updating CSV: {e}")
        import traceback
        traceback.print_exc()
    return 0


async def compare_adaptive_dca_vs_dca(
    symbol: str,
    start_date: date,
    end_date: date,
    weekly_contribution: float = 1000.0,
    initial_cash: float = 100_000.0,
):
    """
    Compare AdaptiveDCA strategy vs Regular DCA week by week.
    
    Prints a column format showing:
    - Date
    - FGI (Fear & Greed Index)
    - AdaptiveDCA Action
    - DCA Action
    """
    print()
    print("=" * 100)
    print("AdaptiveDCA vs Regular DCA - Week by Week Comparison")
    print("=" * 100)
    print()
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Weekly Contribution: ${weekly_contribution:,.2f}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print()
    
    # Import required modules
    from core.data.yfinance_data import YFinanceDataAdapter
    from core.data.fear_greed_index import get_fgi_value
    from datetime import timedelta
    
    # Initialize data engine (yfinance doesn't require API keys)
    data_engine = YFinanceDataAdapter()
    
    # Convert dates to datetime
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    # Get all bars for the date range
    bars = await data_engine.get_bars(symbol, start_dt, end_dt, timeframe="1D")
    if not bars:
        print(f"❌ No data available for {symbol} in date range {start_date} to {end_date}")
        return
    
    # Track weekly actions
    weekly_actions = []
    last_week_key = None
    
    # Process each bar to collect weekly actions
    for bar in bars:
        bar_date = bar.timestamp.date()
        current_week_key = (bar_date.isocalendar()[0], bar_date.isocalendar()[1])
        
        # Check if this is a new week
        is_new_week = (last_week_key is None or current_week_key != last_week_key)
        
        if is_new_week:
            # Get FGI for this date
            fgi_value, fgi_classification = get_fgi_value(target_date=bar_date)
            
            # Determine AdaptiveDCA action
            adaptive_action = "N/A"
            if fgi_value is not None:
                if fgi_value <= 45:
                    # Fear: Buy inversely proportional
                    buy_multiplier = (45 - fgi_value) / 45
                    adaptive_action = f"BUY {buy_multiplier:.1%} of available"
                elif fgi_value <= 75:
                    # Neutral/Greed: Buy percentage of weekly contribution
                    buy_multiplier = (fgi_value - 46) / 29
                    adaptive_action = f"BUY {buy_multiplier:.1%} of ${weekly_contribution:,.0f}"
                else:
                    # Extreme Greed: Sell
                    sell_fraction = 0.5 * (fgi_value - 76) / 24
                    adaptive_action = f"SELL {sell_fraction:.1%} of position"
            
            # DCA always buys 100% of weekly contribution
            dca_action = f"BUY ${weekly_contribution:,.0f}"
            
            weekly_actions.append({
                "date": bar_date,
                "fgi": fgi_value,
                "adaptive_action": adaptive_action,
                "dca_action": dca_action,
            })
            
            last_week_key = current_week_key
    
    # Print column format
    print(f"{'Date':<12} {'FGI':<8} {'AdaptiveDCA Action':<35} {'DCA Action':<20}")
    print("-" * 100)
    
    for action in weekly_actions:
        fgi_str = f"{action['fgi']:.2f}" if action['fgi'] is not None else "N/A"
        print(f"{action['date']:<12} {fgi_str:<8} {action['adaptive_action']:<35} {action['dca_action']:<20}")
    
    print()
    print(f"Total weeks: {len(weekly_actions)}")
    print("=" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(description="Test CNN Fear & Greed Index (current and historical)")
    parser.add_argument(
        "--date",
        type=str,
        help="Test specific historical date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Only test current FGI from package",
    )
    parser.add_argument(
        "--historical-only",
        action="store_true",
        help="Only test historical FGI from CSV",
    )
    parser.add_argument(
        "--update-csv",
        action="store_true",
        help="Fetch missing FGI data from API and append to CSV",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for CSV update (YYYY-MM-DD format, default: CSV min date)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for CSV update (YYYY-MM-DD format, default: today)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare AdaptiveDCA vs Regular DCA week by week",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol for comparison (default: SPY)",
    )
    parser.add_argument(
        "--compare-start",
        type=str,
        help="Start date for comparison (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--compare-end",
        type=str,
        help="End date for comparison (YYYY-MM-DD format, default: today)",
    )
    parser.add_argument(
        "--weekly-contribution",
        type=float,
        default=1000.0,
        help="Weekly contribution amount (default: 1000.0)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CNN Fear & Greed Index Test")
    print("=" * 60)
    print()
    
    # Parse test date if provided
    test_date = None
    if args.date:
        try:
            test_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
            return 1
    
    csv_path = project_root / "core" / "data" / "fear_greed_index.csv"
    
    # Handle comparison mode
    if args.compare:
        import asyncio
        
        if not args.compare_start:
            print("Error: --compare-start is required for comparison mode")
            return 1
        
        try:
            start_date = datetime.strptime(args.compare_start, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid start date format '{args.compare_start}'. Use YYYY-MM-DD format.")
            return 1
        
        if args.compare_end:
            try:
                end_date = datetime.strptime(args.compare_end, "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: Invalid end date format '{args.compare_end}'. Use YYYY-MM-DD format.")
                return 1
        else:
            end_date = date.today()
        
        asyncio.run(compare_adaptive_dca_vs_dca(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            weekly_contribution=args.weekly_contribution,
        ))
        return 0
    
    # Handle CSV update mode
    if args.update_csv:
        start_date = None
        end_date = None
        
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: Invalid start date format '{args.start_date}'. Use YYYY-MM-DD format.")
                return 1
        
        if args.end_date:
            try:
                end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: Invalid end date format '{args.end_date}'. Use YYYY-MM-DD format.")
                return 1
        
        new_count = update_csv_with_missing_data(csv_path, start_date, end_date)
        return 0 if new_count >= 0 else 1
    
    success = True
    
    # Test current FGI
    if not args.historical_only:
        if not test_current_fgi():
            success = False
    
    # Test historical FGI
    if not args.current_only:
        if test_date:
            # Test specific date from API
            print("Testing historical FGI from CNN API...")
            api_success = test_historical_fgi_api(test_date)
            
            # Also test from CSV for comparison
            print("\nComparing with CSV data...")
            csv_success = test_historical_fgi_csv(csv_path, test_date)
            
            if not api_success and not csv_success:
                success = False
        else:
            # Show CSV data samples
            if not test_historical_fgi_csv(csv_path, test_date):
                success = False
    
    # Show interpretation guide
    if not args.current_only and not args.historical_only:
        print("=" * 60)
        print("Interpretation:")
        print("=" * 60)
        print("0-25:   Extreme Fear [RED]")
        print("26-45:  Fear [ORANGE]")
        print("46-55:  Neutral [YELLOW]")
        print("56-75:  Greed [GREEN]")
        print("76-100: Extreme Greed [GREEN++]")
        print()
    
    print("=" * 60)
    if success:
        print("Test completed successfully!")
    else:
        print("Test completed with warnings/errors")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
