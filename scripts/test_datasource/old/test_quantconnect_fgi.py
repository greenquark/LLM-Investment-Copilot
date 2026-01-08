"""
Test script to pull Fear and Greed Index data from QuantConnect for random dates from July 2014.

This script:
1. Generates random dates from July 2014 onwards
2. Attempts to fetch Fear and Greed data from QuantConnect API
3. Compares with existing data sources (CNN API, CSV)
4. Prints results and statistics

Reference: https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/fear-and-greed
"""

import sys
import random
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Dict
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    print("[WARN] httpx not available, API calls will be skipped")

from core.data.fear_greed_index import get_fgi_value, _get_historical_fgi_from_api
from core.data import is_trading_day


def generate_random_dates(
    start_date: date,
    end_date: date,
    num_dates: int,
    only_trading_days: bool = True,
) -> List[date]:
    """
    Generate random dates within a given range.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        num_dates: Number of random dates to generate
        only_trading_days: If True, only generate trading days
    
    Returns:
        List of random dates
    """
    date_list = []
    total_days = (end_date - start_date).days + 1
    
    # If we need trading days, first get all trading days in range
    if only_trading_days:
        trading_days = []
        current = start_date
        while current <= end_date:
            try:
                if is_trading_day(current):
                    trading_days.append(current)
            except Exception:
                # If trading day check fails, include the date anyway
                trading_days.append(current)
            current += timedelta(days=1)
        
        if len(trading_days) < num_dates:
            print(f"[WARN] Only {len(trading_days)} trading days available, generating {len(trading_days)} dates")
            return random.sample(trading_days, len(trading_days))
        
        return random.sample(trading_days, num_dates)
    else:
        # Generate random dates (may include weekends/holidays)
        for _ in range(num_dates):
            random_days = random.randint(0, total_days - 1)
            random_date = start_date + timedelta(days=random_days)
            date_list.append(random_date)
        return date_list


def fetch_quantconnect_fgi(
    target_date: date,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> Optional[Tuple[float, Optional[str]]]:
    """
    Fetch Fear and Greed Index data from QuantConnect API.
    
    Note: QuantConnect's API typically requires authentication.
    This function attempts to use their REST API endpoints.
    
    Args:
        target_date: Date to fetch FGI for
        api_key: QuantConnect API key (optional)
        user_id: QuantConnect user ID (optional)
        api_secret: QuantConnect API secret (optional)
    
    Returns:
        Tuple of (value, classification) or None if unavailable
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not _HTTPX_AVAILABLE:
        return None
    
    date_str = target_date.strftime("%Y-%m-%d")
    
    # QuantConnect API endpoints (may vary - check their documentation)
    # Option 1: Direct data API (if available)
    base_url = "https://www.quantconnect.com/api/v2"
    
    # Try different possible endpoints
    endpoints = [
        f"{base_url}/data/fear-and-greed",
        f"{base_url}/data/feargreed",
        f"{base_url}/datasets/fear-and-greed",
    ]
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    if user_id:
        headers["X-User-Id"] = user_id
    
    for endpoint in endpoints:
        try:
            url = f"{endpoint}?date={date_str}"
            if api_secret:
                # Some APIs use different auth methods
                pass
            
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Parse response (structure may vary)
                        if isinstance(data, dict):
                            if "value" in data:
                                value = float(data["value"])
                                classification = data.get("classification") or data.get("rating")
                                return value, classification
                            elif "fearGreed" in data:
                                value = float(data["fearGreed"])
                                return value, None
                    except Exception as e:
                        logger.debug(f"Failed to parse QuantConnect response: {e}")
                        continue
                elif response.status_code == 401:
                    print(f"[WARN] QuantConnect API authentication required for {date_str}")
                    break
                elif response.status_code == 404:
                    # Try next endpoint
                    continue
        except Exception as e:
            logger.debug(f"QuantConnect API error for {endpoint}: {e}")
            continue
    
    return None


def test_quantconnect_fgi(
    num_dates: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
    api_secret: Optional[str] = None,
    compare_with_cnn: bool = True,
) -> None:
    """
    Test fetching Fear and Greed data from QuantConnect for random dates.
    
    Args:
        num_dates: Number of random dates to test
        start_date: Start date (default: 2014-07-01)
        end_date: End date (default: today)
        api_key: QuantConnect API key (optional)
        user_id: QuantConnect user ID (optional)
        api_secret: QuantConnect API secret (optional)
        compare_with_cnn: If True, compare with CNN API data
    """
    import logging
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("QuantConnect Fear & Greed Index Data Test")
    print("=" * 80)
    print()
    
    # Set default dates
    if start_date is None:
        start_date = date(2014, 7, 1)
    if end_date is None:
        end_date = date.today()
    
    print(f"[INFO] Testing {num_dates} random dates from {start_date} to {end_date}")
    print(f"[INFO] QuantConnect API authentication: {'Provided' if api_key else 'Not provided (may fail)'}")
    print()
    
    # Generate random dates
    print(f"[INFO] Generating {num_dates} random trading dates...")
    random_dates = generate_random_dates(start_date, end_date, num_dates, only_trading_days=True)
    random_dates.sort()  # Sort for easier reading
    print(f"[OK] Generated {len(random_dates)} dates")
    print()
    
    # Test results
    results: List[Dict] = []
    qc_success = 0
    qc_failed = 0
    cnn_success = 0
    cnn_failed = 0
    matches = 0
    mismatches = 0
    
    print("=" * 80)
    print("FETCHING DATA")
    print("=" * 80)
    print()
    
    for i, test_date in enumerate(random_dates, 1):
        print(f"[{i}/{len(random_dates)}] Testing {test_date}...")
        
        # Fetch from QuantConnect
        qc_result = fetch_quantconnect_fgi(
            test_date,
            api_key=api_key,
            user_id=user_id,
            api_secret=api_secret,
        )
        
        if qc_result is not None:
            qc_value, qc_classification = qc_result
        else:
            qc_value, qc_classification = None, None
        
        # Fetch from CNN API for comparison
        cnn_value, cnn_classification = None, None
        if compare_with_cnn:
            cnn_result = _get_historical_fgi_from_api(test_date)
            if cnn_result is not None:
                cnn_value, cnn_classification = cnn_result
        
        # Store result
        result = {
            "date": test_date,
            "qc_value": qc_value,
            "qc_classification": qc_classification,
            "cnn_value": cnn_value,
            "cnn_classification": cnn_classification,
        }
        results.append(result)
        
        # Update statistics
        if qc_value is not None:
            qc_success += 1
            print(f"  QuantConnect: {qc_value:.2f} ({qc_classification or 'N/A'})")
        else:
            qc_failed += 1
            print(f"  QuantConnect: Failed/No data")
        
        if cnn_value is not None:
            cnn_success += 1
            print(f"  CNN API: {cnn_value:.2f} ({cnn_classification or 'N/A'})")
            
            # Compare if both succeeded
            if qc_value is not None:
                diff = abs(qc_value - cnn_value)
                if diff < 0.01:  # Values match (within tolerance)
                    matches += 1
                    print(f"  Match: Values are identical (diff: {diff:.4f})")
                else:
                    mismatches += 1
                    print(f"  Mismatch: Difference = {diff:.2f}")
        else:
            cnn_failed += 1
            print(f"  CNN API: Failed/No data")
        
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total dates tested: {len(random_dates)}")
    print()
    print("QuantConnect API:")
    print(f"  Success: {qc_success}")
    print(f"  Failed: {qc_failed}")
    print(f"  Success rate: {qc_success/len(random_dates)*100:.1f}%")
    print()
    
    if compare_with_cnn:
        print("CNN API:")
        print(f"  Success: {cnn_success}")
        print(f"  Failed: {cnn_failed}")
        print(f"  Success rate: {cnn_success/len(random_dates)*100:.1f}%")
        print()
        
        if qc_success > 0 and cnn_success > 0:
            print("Comparison:")
            print(f"  Matches: {matches}")
            print(f"  Mismatches: {mismatches}")
            if matches + mismatches > 0:
                print(f"  Match rate: {matches/(matches+mismatches)*100:.1f}%")
            print()
    
    # Print detailed results table
    print("=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    print(f"{'Date':<12} {'QuantConnect':<20} {'CNN API':<20} {'Status':<15}")
    print("-" * 80)
    
    for result in results:
        date_str = str(result["date"])
        qc_str = f"{result['qc_value']:.2f}" if result["qc_value"] is not None else "N/A"
        cnn_str = f"{result['cnn_value']:.2f}" if result["cnn_value"] is not None else "N/A"
        
        if result["qc_value"] is not None and result["cnn_value"] is not None:
            diff = abs(result["qc_value"] - result["cnn_value"])
            if diff < 0.01:
                status = "Match"
            else:
                status = f"Mismatch ({diff:.2f})"
        elif result["qc_value"] is not None:
            status = "QC only"
        elif result["cnn_value"] is not None:
            status = "CNN only"
        else:
            status = "Both failed"
        
        print(f"{date_str:<12} {qc_str:<20} {cnn_str:<20} {status:<15}")
    
    print()
    print("=" * 80)
    
    # Notes
    print()
    print("NOTES:")
    print("- QuantConnect API may require authentication (API key, user ID, etc.)")
    print("- If authentication fails, you may need to:")
    print("  1. Sign up for a QuantConnect account")
    print("  2. Generate an API key from your account settings")
    print("  3. Pass the API key using --api-key argument")
    print("- QuantConnect's API endpoints may differ from what's implemented here")
    print("- Refer to QuantConnect API documentation for the correct endpoints")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test QuantConnect Fear and Greed Index data fetching"
    )
    parser.add_argument(
        "--num-dates",
        type=int,
        default=20,
        help="Number of random dates to test (default: 20)"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD, default: 2014-07-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="QuantConnect API key (optional)"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="QuantConnect user ID (optional)"
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        help="QuantConnect API secret (optional)"
    )
    parser.add_argument(
        "--no-cnn-compare",
        action="store_true",
        help="Don't compare with CNN API"
    )
    
    args = parser.parse_args()
    
    start_date = None
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    
    end_date = None
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    test_quantconnect_fgi(
        num_dates=args.num_dates,
        start_date=start_date,
        end_date=end_date,
        api_key=args.api_key,
        user_id=args.user_id,
        api_secret=args.api_secret,
        compare_with_cnn=not args.no_cnn_compare,
    )

