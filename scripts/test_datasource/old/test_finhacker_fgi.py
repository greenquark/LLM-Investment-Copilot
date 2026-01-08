"""
Test script to fetch Fear and Greed Index data from finhacker.cz.

Reference: https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/

This script attempts to:
1. Find the API endpoint used by the interactive chart
2. Download historical data if available
3. Parse and compare with existing data sources
"""

import sys
import json
import random
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List, Dict
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    from bs4 import BeautifulSoup
    _HTTPX_AVAILABLE = True
    _BS4_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    _BS4_AVAILABLE = False
    print("[WARN] httpx or BeautifulSoup4 not available")

from core.data.fear_greed_index import _get_historical_fgi_from_api


def inspect_finhacker_page() -> Optional[Dict]:
    """
    Inspect the finhacker.cz page to find how data is loaded.
    
    Returns:
        Dictionary with found endpoints or data structure, or None
    """
    if not _HTTPX_AVAILABLE:
        print("[ERROR] httpx not available")
        return None
    
    url = "https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/"
    
    print(f"[INFO] Inspecting page: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            
            html = response.text
            print(f"[OK] Page loaded ({len(html)} bytes)")
            
            # Try to find API endpoints or data in the HTML
            findings = {}
            
            # Look for common API patterns
            if "api" in html.lower():
                print("[INFO] Found 'api' in page content")
            
            # Look for JSON data embedded in script tags
            if _BS4_AVAILABLE:
                soup = BeautifulSoup(html, 'html.parser')
                scripts = soup.find_all('script')
                
                for script in scripts:
                    script_text = script.string or ""
                    # Look for JSON data or API endpoints
                    if "fear" in script_text.lower() and "greed" in script_text.lower():
                        # Try to extract JSON
                        try:
                            # Look for JSON objects
                            if "{" in script_text and "}" in script_text:
                                # Try to find data arrays
                                print("[INFO] Found potential JSON data in script tag")
                        except:
                            pass
                
                # Look for data attributes
                chart_elements = soup.find_all(attrs={"data-chart": True}) or soup.find_all(class_="chart")
                if chart_elements:
                    print(f"[INFO] Found {len(chart_elements)} chart elements")
            
            # Look for fetch/axios/XMLHttpRequest calls
            if "fetch(" in html or "axios" in html.lower() or "XMLHttpRequest" in html:
                print("[INFO] Page likely uses JavaScript to load data dynamically")
                print("[INFO] May need to inspect network requests or use browser automation")
            
            return findings
            
    except Exception as e:
        print(f"[ERROR] Failed to inspect page: {e}")
        return None


def fetch_finhacker_fgi_via_api(
    target_date: date,
    base_url: str = "https://www.finhacker.cz"
) -> Optional[Tuple[float, Optional[str]]]:
    """
    Attempt to fetch FGI data from finhacker.cz API endpoints.
    
    Note: This function tries common API endpoint patterns.
    The actual endpoint may need to be discovered by inspecting network requests.
    
    Args:
        target_date: Date to fetch FGI for
        base_url: Base URL for finhacker.cz
    
    Returns:
        Tuple of (value, classification) or None if unavailable
    """
    if not _HTTPX_AVAILABLE:
        return None
    
    date_str = target_date.strftime("%Y-%m-%d")
    
    # Possible API endpoints (to be discovered)
    possible_endpoints = [
        f"{base_url}/api/fear-greed-index",
        f"{base_url}/api/v1/fear-greed-index",
        f"{base_url}/api/data/fear-greed-index",
        f"{base_url}/api/fgi",
        f"{base_url}/data/fear-greed-index.json",
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    
    for endpoint in possible_endpoints:
        try:
            # Try with date parameter
            url = f"{endpoint}?date={date_str}"
            
            with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Parse response
                        if isinstance(data, dict):
                            if "value" in data or "fgi" in data or "fearGreed" in data:
                                value = float(data.get("value") or data.get("fgi") or data.get("fearGreed"))
                                classification = data.get("classification") or data.get("rating")
                                return value, classification
                    except:
                        pass
        except Exception:
            continue
    
    return None


def fetch_finhacker_fgi_via_scraping(
    target_date: date,
    url: str = "https://www.finhacker.cz/en/fear-and-greed-index-historical-data-and-chart/"
) -> Optional[Tuple[float, Optional[str]]]:
    """
    Attempt to scrape FGI data from finhacker.cz page.
    
    This is a fallback method if no API is available.
    Note: Web scraping may be fragile and should respect robots.txt and rate limits.
    
    Args:
        target_date: Date to fetch FGI for
        url: URL of the finhacker.cz page
    
    Returns:
        Tuple of (value, classification) or None if unavailable
    """
    if not _HTTPX_AVAILABLE or not _BS4_AVAILABLE:
        return None
    
    # This would require parsing the page structure
    # For now, return None as we need to inspect the actual page structure
    return None


def download_finhacker_data(
    output_file: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[Dict]:
    """
    Attempt to download historical FGI data from finhacker.cz.
    
    This function tries to:
    1. Find a download/export endpoint
    2. Download the full dataset
    3. Parse and save to file
    
    Args:
        output_file: Optional path to save downloaded data
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        Dictionary of date -> (value, classification) or None
    """
    if not _HTTPX_AVAILABLE:
        print("[ERROR] httpx not available")
        return None
    
    base_url = "https://www.finhacker.cz"
    
    # Possible download endpoints
    download_endpoints = [
        f"{base_url}/api/fear-greed-index/download",
        f"{base_url}/api/fear-greed-index/export",
        f"{base_url}/data/fear-greed-index.csv",
        f"{base_url}/data/fear-greed-index.json",
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/csv, */*",
    }
    
    for endpoint in download_endpoints:
        try:
            with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
                response = client.get(endpoint)
                
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    
                    if "json" in content_type:
                        data = response.json()
                        # Parse JSON data
                        fgi_data = {}
                        # Structure depends on API response
                        # This is a placeholder - actual parsing depends on API structure
                        print(f"[OK] Downloaded JSON data from {endpoint}")
                        return fgi_data
                    elif "csv" in content_type or endpoint.endswith(".csv"):
                        # Parse CSV
                        import pandas as pd
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))
                        print(f"[OK] Downloaded CSV data from {endpoint}")
                        # Parse CSV structure
                        fgi_data = {}
                        # This is a placeholder - actual parsing depends on CSV structure
                        return fgi_data
        except Exception as e:
            print(f"[DEBUG] Endpoint {endpoint} failed: {e}")
            continue
    
    print("[WARN] No download endpoint found")
    return None


def test_finhacker_fgi(
    num_dates: int = 10,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    compare_with_cnn: bool = True,
) -> None:
    """
    Test fetching FGI data from finhacker.cz.
    
    Args:
        num_dates: Number of random dates to test
        start_date: Start date (default: 2014-07-01)
        end_date: End date (default: today)
        compare_with_cnn: If True, compare with CNN API data
    """
    import random
    
    print("=" * 80)
    print("FinHacker.cz Fear & Greed Index Data Test")
    print("=" * 80)
    print()
    
    # Set default dates
    if start_date is None:
        start_date = date(2014, 7, 1)
    if end_date is None:
        end_date = date.today()
    
    print(f"[INFO] Testing data fetching from finhacker.cz")
    print(f"[INFO] Date range: {start_date} to {end_date}")
    print()
    
    # Step 1: Inspect the page
    print("=" * 80)
    print("STEP 1: Inspecting finhacker.cz page structure")
    print("=" * 80)
    print()
    findings = inspect_finhacker_page()
    print()
    
    # Step 2: Try to download full dataset
    print("=" * 80)
    print("STEP 2: Attempting to download historical data")
    print("=" * 80)
    print()
    downloaded_data = download_finhacker_data()
    if downloaded_data:
        print(f"[OK] Downloaded {len(downloaded_data)} records")
    else:
        print("[INFO] No download endpoint found - may need manual inspection")
    print()
    
    # Step 3: Test individual date fetching
    print("=" * 80)
    print("STEP 3: Testing individual date fetching")
    print("=" * 80)
    print()
    
    # Generate test dates
    total_days = (end_date - start_date).days
    test_dates = []
    for _ in range(num_dates):
        random_days = random.randint(0, total_days)
        test_date = start_date + timedelta(days=random_days)
        test_dates.append(test_date)
    test_dates.sort()
    
    results = []
    for test_date in test_dates:
        print(f"Testing {test_date}...")
        
        # Try API
        fh_result = fetch_finhacker_fgi_via_api(test_date)
        if fh_result:
            fh_value, fh_class = fh_result
            print(f"  FinHacker API: {fh_value:.2f} ({fh_class or 'N/A'})")
        else:
            fh_value, fh_class = None, None
            print(f"  FinHacker API: Failed/No endpoint found")
        
        # Compare with CNN
        cnn_value, cnn_class = None, None
        if compare_with_cnn:
            cnn_result = _get_historical_fgi_from_api(test_date)
            if cnn_result is not None:
                cnn_value, cnn_class = cnn_result
                if cnn_value is not None:
                    print(f"  CNN API: {cnn_value:.2f} ({cnn_class or 'N/A'})")
                else:
                    print(f"  CNN API: Failed/No data")
            else:
                print(f"  CNN API: Failed/No data")
        
        results.append({
            "date": test_date,
            "finhacker_value": fh_value,
            "finhacker_class": fh_class,
            "cnn_value": cnn_value,
            "cnn_class": cnn_class,
        })
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("RECOMMENDATIONS:")
    print("1. Inspect the finhacker.cz page in a browser with Developer Tools open")
    print("2. Look at Network tab to find API endpoints when the chart loads")
    print("3. Check if there's a 'Download chart' button that provides data export")
    print("4. The page may use JavaScript to load data - may need Selenium/Playwright")
    print("5. Consider contacting finhacker.cz to ask about API access")
    print()
    print("ALTERNATIVE APPROACHES:")
    print("- Use browser automation (Selenium/Playwright) to interact with the page")
    print("- Check if they provide a CSV/JSON download link")
    print("- Look for embedded JSON data in the page source")
    print("- Use the Wayback Machine to get historical snapshots")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test fetching Fear and Greed Index data from finhacker.cz"
    )
    parser.add_argument(
        "--num-dates",
        type=int,
        default=10,
        help="Number of test dates (default: 10)"
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
    
    test_finhacker_fgi(
        num_dates=args.num_dates,
        start_date=start_date,
        end_date=end_date,
        compare_with_cnn=not args.no_cnn_compare,
    )

