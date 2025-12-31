"""
Test cache program for pseudo quotes.

This program:
1. Generates random dates for ticker FAKE from July 1 to Oct 31, 2025
2. Creates pseudo quotes for those dates
3. Saves them to cache
4. Tests existence
5. Pulls/retrieves them
6. Verifies 100% coverage
"""

import asyncio
import sys
import os
import random
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Set
import pandas as pd
import pyarrow.parquet as pq

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.bar import Bar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class QuoteCache:
    """Simple quote cache for testing purposes (uses Bar model)."""
    
    def __init__(self, cache_dir: str = "data_cache/bars"):
        """Initialize the quote cache (uses bars directory)."""
        # Resolve cache directory relative to project root if it's a relative path
        cache_dir_path = Path(cache_dir)
        if not cache_dir_path.is_absolute():
            # Make it relative to project root
            self._cache_dir = project_root / cache_dir
        else:
            self._cache_dir = cache_dir_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"QuoteCache initialized with cache directory: {self._cache_dir}")
    
    def get_cache_path(self, symbol: str) -> Path:
        """Get the cache file path for a symbol (uses same format as bar cache)."""
        normalized_symbol = symbol.upper()
        # Use same naming convention as bar cache: SYMBOL_TIMEFRAME.parquet
        cache_filename = f"{normalized_symbol}_1D.parquet"
        return self._cache_dir / cache_filename
    
    def purge_cache(self, symbol: str) -> bool:
        """Delete the cache file for a symbol. Returns True if deleted, False if not found."""
        cache_path = self.get_cache_path(symbol)
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Purged cache file: {cache_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to purge cache file {cache_path}: {e}")
                return False
        else:
            logger.info(f"Cache file does not exist: {cache_path}")
            return False
    
    async def save_quotes(
        self,
        symbol: str,
        quotes: List[Bar],
    ) -> None:
        """Save quotes to cache, merging with existing data if present."""
        if not quotes:
            return
        
        cache_path = self.get_cache_path(symbol)
        
        try:
            # Load existing cache if it exists
            existing_quotes: List[Bar] = []
            if cache_path.exists():
                try:
                    df_existing = await asyncio.to_thread(pd.read_parquet, cache_path)
                    if 'timestamp' in df_existing.columns:
                        df_existing = df_existing.set_index('timestamp')
                    df_existing.index = pd.to_datetime(df_existing.index)
                    
                    # Convert existing to Bar objects
                    for idx, row in df_existing.iterrows():
                        existing_quotes.append(
                            Bar(
                                symbol=symbol,
                                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                                open=float(row['open']),
                                high=float(row['high']),
                                low=float(row['low']),
                                close=float(row['close']),
                                volume=float(row['volume']),
                                timeframe=row.get('timeframe', '1D'),
                            )
                        )
                    logger.debug(f"Loaded from file: {len(existing_quotes)} quotes")
                except Exception as e:
                    logger.warning(f"Failed to load existing cache for merging: {e}")
                    existing_quotes = []
            
            # Merge and deduplicate
            merged_quotes = self._merge_and_deduplicate(existing_quotes, quotes)
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': q.timestamp,
                    'open': q.open,
                    'high': q.high,
                    'low': q.low,
                    'close': q.close,
                    'volume': q.volume,
                    'timeframe': q.timeframe,
                    'symbol': q.symbol,
                }
                for q in merged_quotes
            ])
            
            # Set timestamp as index and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Write to temporary file first (atomic write)
            temp_path = cache_path.with_suffix('.parquet.tmp')
            
            # Remove temp file if it exists from a previous failed write
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            
            await asyncio.to_thread(
                df.to_parquet,
                temp_path,
                engine='pyarrow',
                compression='snappy',
                index=True,
            )
            
            # Atomic rename
            temp_path.replace(cache_path)
            
            actual_new = len(merged_quotes) - len(existing_quotes)
            logger.info(
                f"Saved {len(merged_quotes)} quotes to cache: {cache_path} "
                f"(added {actual_new} new, had {len(existing_quotes)} existing, "
                f"received {len(quotes)} from caller)"
            )
            
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            raise
    
    def _merge_and_deduplicate(
        self,
        existing: List[Bar],
        new: List[Bar],
    ) -> List[Bar]:
        """Merge cached and new quotes (bars), handling overlaps intelligently."""
        # Create a dict keyed by timestamp for fast lookup
        merged_dict = {}
        
        # First, add all existing bars
        for bar in existing:
            # Normalize timestamp to seconds (remove microseconds) for comparison
            normalized_ts = bar.timestamp.replace(microsecond=0)
            key = (normalized_ts, bar.symbol, bar.timeframe)
            merged_dict[key] = bar
        
        # Then add new bars, but don't overwrite existing ones
        for bar in new:
            normalized_ts = bar.timestamp.replace(microsecond=0)
            key = (normalized_ts, bar.symbol, bar.timeframe)
            if key not in merged_dict:
                merged_dict[key] = bar
        
        # Convert back to list and sort by timestamp
        merged = list(merged_dict.values())
        merged.sort(key=lambda b: b.timestamp)
        
        return merged
    
    async def load_cached_quotes(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[List[Bar]]:
        """Load cached quotes (bars), optionally filtered by date range."""
        cache_path = self.get_cache_path(symbol)
        
        if not cache_path.exists():
            return None
        
        try:
            def _read_all():
                df = pd.read_parquet(cache_path)
                
                # Ensure timestamp is datetime and set as index if not already
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif df.index.name != 'timestamp':
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'timestamp'
                
                if df.empty:
                    return []
                
                # Convert to Bar objects
                bars = []
                for idx, row in df.iterrows():
                    bars.append(
                        Bar(
                            symbol=symbol,
                            timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume']),
                            timeframe=row.get('timeframe', '1D'),
                        )
                    )
                return bars
            
            all_quotes = await asyncio.to_thread(_read_all)
            
            if not all_quotes:
                return None
            
            # Filter by date range if provided
            if start or end:
                filtered = []
                for bar in all_quotes:
                    if start and bar.timestamp < start:
                        continue
                    if end and bar.timestamp > end:
                        continue
                    filtered.append(bar)
                return filtered if filtered else None
            
            return all_quotes
            
        except Exception as e:
            logger.warning(
                f"Failed to load cache from {cache_path}: {e}. "
                "Will return None."
            )
            return None
    
    def quote_exists(
        self,
        symbol: str,
        quote_timestamp: datetime,
    ) -> bool:
        """Check if a quote (bar) exists in cache for a specific timestamp."""
        cache_path = self.get_cache_path(symbol)
        
        if not cache_path.exists():
            return False
        
        try:
            # Load all bars and check if timestamp exists
            df = pd.read_parquet(cache_path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
            
            # Normalize timestamp for comparison
            normalized_ts = quote_timestamp.replace(microsecond=0)
            df.index = df.index.map(lambda x: x.replace(microsecond=0))
            
            return normalized_ts in df.index
            
        except Exception as e:
            logger.warning(f"Failed to check quote existence: {e}")
            return False
    
    def get_all_cached_dates(
        self,
        symbol: str,
    ) -> Set[date]:
        """Get all dates that have cached quotes."""
        cache_path = self.get_cache_path(symbol)
        
        if not cache_path.exists():
            return set()
        
        try:
            df = pd.read_parquet(cache_path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
            
            # Extract unique dates
            dates = set(df.index.date)
            return dates
            
        except Exception as e:
            logger.warning(f"Failed to get cached dates: {e}")
            return set()


def generate_random_dates(start_date: date, end_date: date, count: int) -> List[date]:
    """Generate random dates within the specified range."""
    dates = []
    delta = (end_date - start_date).days
    
    for _ in range(count):
        random_days = random.randint(0, delta)
        random_date = start_date + timedelta(days=random_days)
        dates.append(random_date)
    
    # Remove duplicates and sort
    dates = sorted(list(set(dates)))
    return dates


def create_pseudo_quote(symbol: str, quote_date: date, base_price: float = 100.0) -> Bar:
    """Create a pseudo quote (as a Bar) for testing."""
    # Generate random price around base_price
    price_variation = random.uniform(-10, 10)
    close_price = base_price + price_variation
    
    # Generate OHLC from close price
    high = close_price + random.uniform(0, 5)
    low = close_price - random.uniform(0, 5)
    open_price = random.uniform(low, high)
    
    # Generate volume
    volume = random.uniform(1000, 100000)
    
    # Create timestamp at market open (9:30 AM)
    quote_timestamp = datetime.combine(quote_date, datetime.min.time().replace(hour=9, minute=30))
    
    return Bar(
        symbol=symbol,
        timestamp=quote_timestamp,
        open=round(open_price, 2),
        high=round(high, 2),
        low=round(low, 2),
        close=round(close_price, 2),
        volume=volume,
        timeframe='1D',
    )


async def main():
    """Main test function."""
    ticker = "FAKE"
    start_date = date(2025, 7, 1)
    end_date = date(2025, 10, 31)
    
    logger.info(f"Starting quote cache test for {ticker}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Initialize cache (use same directory as bars)
    cache = QuoteCache(cache_dir="data_cache/bars")
    
    # Step 0: Purge and delete FAKE cache at the beginning
    logger.info(f"Purging existing cache for {ticker}...")
    cache.purge_cache(ticker)
    logger.info("Cache purged successfully")
    
    # Step 1: Generate random dates
    # First, generate quotes for all dates in the range to ensure 100% coverage
    # Then also generate some random dates to test the system
    total_days = (end_date - start_date).days + 1
    
    # Generate quotes for ALL dates in the range (for 100% coverage)
    all_dates = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)
    
    # Also generate some additional random dates (may overlap, which is fine)
    # This tests the deduplication logic
    num_random_quotes = int(total_days * 0.3)  # 30% additional random dates
    logger.info(f"Generating {num_random_quotes} additional random dates...")
    random_dates = generate_random_dates(start_date, end_date, num_random_quotes)
    
    # Combine all dates (set will remove duplicates)
    all_dates_set = set(all_dates) | set(random_dates)
    all_dates = sorted(list(all_dates_set))
    
    logger.info(f"Generated {len(all_dates)} unique dates (all dates in range + random)")
    logger.info(f"Date range: {min(all_dates)} to {max(all_dates)}")
    
    # Step 2: Create pseudo quotes for those dates
    logger.info(f"Creating pseudo quotes for {len(all_dates)} dates...")
    quotes = []
    for quote_date in all_dates:
        quote = create_pseudo_quote(ticker, quote_date)
        quotes.append(quote)
    
    logger.info(f"Created {len(quotes)} pseudo quotes")
    
    # Step 3: Save quotes to cache
    logger.info(f"Saving {len(quotes)} quotes to cache...")
    await cache.save_quotes(ticker, quotes)
    logger.info("Quotes saved to cache")
    
    # Step 4: Test existence for all saved dates
    logger.info("Testing existence of saved quotes...")
    existing_count = 0
    missing_count = 0
    for quote_date in all_dates:
        quote_timestamp = datetime.combine(quote_date, datetime.min.time().replace(hour=9, minute=30))
        exists = cache.quote_exists(ticker, quote_timestamp)
        if exists:
            existing_count += 1
        else:
            missing_count += 1
            logger.warning(f"Quote missing for date: {quote_date}")
    
    logger.info(f"Existence test: {existing_count} found, {missing_count} missing")
    
    # Step 5: Pull/retrieve quotes
    logger.info("Pulling quotes from cache...")
    retrieved_quotes = await cache.load_cached_quotes(
        ticker,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
    )
    
    if retrieved_quotes:
        logger.info(f"Retrieved {len(retrieved_quotes)} quotes from cache")
        retrieved_dates = {q.timestamp.date() for q in retrieved_quotes}
        logger.info(f"Retrieved quotes cover {len(retrieved_dates)} unique dates")
    else:
        logger.error("Failed to retrieve quotes from cache!")
        return
    
    # Step 6: Verify 100% coverage
    logger.info("Verifying 100% coverage...")
    
    # Get all cached dates
    cached_dates = cache.get_all_cached_dates(ticker)
    logger.info(f"Total cached dates: {len(cached_dates)}")
    
    # Check coverage for the requested range
    requested_dates = set()
    current = start_date
    while current <= end_date:
        requested_dates.add(current)
        current += timedelta(days=1)
    
    logger.info(f"Total dates in requested range: {len(requested_dates)}")
    
    # Check which dates are covered
    covered_dates = cached_dates.intersection(requested_dates)
    missing_dates = requested_dates - cached_dates
    
    coverage_pct = (len(covered_dates) / len(requested_dates)) * 100 if requested_dates else 0
    
    logger.info(f"Coverage: {len(covered_dates)}/{len(requested_dates)} dates ({coverage_pct:.2f}%)")
    
    if missing_dates:
        logger.warning(f"Missing dates: {sorted(list(missing_dates))[:10]}...")  # Show first 10
        logger.warning(f"Total missing: {len(missing_dates)} dates")
    else:
        logger.info("[SUCCESS] 100% coverage achieved!")
    
    # Additional verification: Check that all saved quotes can be retrieved
    logger.info("Verifying all saved quotes can be retrieved...")
    saved_dates = {q.timestamp.date() for q in quotes}
    retrieved_dates_set = {q.timestamp.date() for q in retrieved_quotes}
    
    if saved_dates == retrieved_dates_set:
        logger.info("[SUCCESS] All saved quotes were successfully retrieved!")
    else:
        missing_in_retrieval = saved_dates - retrieved_dates_set
        extra_in_retrieval = retrieved_dates_set - saved_dates
        if missing_in_retrieval:
            logger.warning(f"Missing in retrieval: {sorted(list(missing_in_retrieval))}")
        if extra_in_retrieval:
            logger.warning(f"Extra in retrieval: {sorted(list(extra_in_retrieval))}")
    
    # Step 7: Verify value matching - compare all fields
    logger.info("Verifying value matching after loading from cache...")
    
    # Create a lookup dictionary for original quotes by timestamp
    original_quotes_dict = {}
    for quote in quotes:
        # Normalize timestamp to seconds for comparison
        normalized_ts = quote.timestamp.replace(microsecond=0)
        original_quotes_dict[normalized_ts] = quote
    
    # Create a lookup dictionary for retrieved quotes by timestamp
    retrieved_quotes_dict = {}
    for quote in retrieved_quotes:
        normalized_ts = quote.timestamp.replace(microsecond=0)
        retrieved_quotes_dict[normalized_ts] = quote
    
    # Compare values
    value_mismatches = []
    all_matched = True
    
    for normalized_ts, original_quote in original_quotes_dict.items():
        if normalized_ts not in retrieved_quotes_dict:
            value_mismatches.append({
                'timestamp': original_quote.timestamp,
                'issue': 'Missing in retrieved quotes',
                'original': original_quote,
                'retrieved': None,
            })
            all_matched = False
            continue
        
        retrieved_quote = retrieved_quotes_dict[normalized_ts]
        
        # Compare all fields
        fields_to_check = ['open', 'high', 'low', 'close', 'volume', 'timeframe']
        mismatched_fields = []
        
        for field in fields_to_check:
            orig_val = getattr(original_quote, field)
            retr_val = getattr(retrieved_quote, field)
            
            # For float comparison, use small epsilon
            if isinstance(orig_val, float) and isinstance(retr_val, float):
                if abs(orig_val - retr_val) > 0.01:  # Allow 0.01 difference for rounding
                    mismatched_fields.append({
                        'field': field,
                        'original': orig_val,
                        'retrieved': retr_val,
                        'diff': abs(orig_val - retr_val),
                    })
            elif orig_val != retr_val:
                mismatched_fields.append({
                    'field': field,
                    'original': orig_val,
                    'retrieved': retr_val,
                })
        
        if mismatched_fields:
            value_mismatches.append({
                'timestamp': original_quote.timestamp,
                'issue': 'Value mismatch',
                'original': original_quote,
                'retrieved': retrieved_quote,
                'mismatched_fields': mismatched_fields,
            })
            all_matched = False
    
    if all_matched:
        logger.info("[SUCCESS] All values match perfectly after loading from cache!")
    else:
        logger.warning(f"[WARNING] Found {len(value_mismatches)} value mismatches")
        for mismatch in value_mismatches[:5]:  # Show first 5
            logger.warning(f"Mismatch at {mismatch['timestamp']}: {mismatch['issue']}")
            if 'mismatched_fields' in mismatch:
                for field_mismatch in mismatch['mismatched_fields']:
                    logger.warning(f"  {field_mismatch['field']}: original={field_mismatch['original']}, retrieved={field_mismatch['retrieved']}")
    
    # Final summary
    print("\n" + "="*60)
    print("QUOTE CACHE TEST SUMMARY")
    print("="*60)
    print(f"Ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Total Days in Range: {len(requested_dates)}")
    print(f"Quotes Generated: {len(quotes)}")
    print(f"Unique Dates with Quotes: {len(saved_dates)}")
    print(f"Quotes Retrieved: {len(retrieved_quotes)}")
    print(f"Coverage: {len(covered_dates)}/{len(requested_dates)} ({coverage_pct:.2f}%)")
    print(f"Existence Test: {existing_count} found, {missing_count} missing")
    print(f"Value Matching: {'PASSED' if all_matched else f'FAILED ({len(value_mismatches)} mismatches)'}")
    
    if coverage_pct == 100.0:
        print("[SUCCESS] 100% COVERAGE VERIFIED!")
    else:
        print(f"[WARNING] Coverage: {coverage_pct:.2f}% (not 100%)")
    
    print("="*60)
    
    # Print all values for visual inspection
    print("\n" + "="*120)
    print("VISUAL INSPECTION: ALL ORIGINAL vs RETRIEVED VALUES")
    print("="*120)
    print(f"{'Date':<12} {'Time':<10} {'Field':<8} {'Original':<15} {'Retrieved':<15} {'Match':<8}")
    print("-" * 120)
    
    # Sort quotes by timestamp for easier comparison
    sorted_original = sorted(quotes, key=lambda q: q.timestamp)
    sorted_retrieved = sorted(retrieved_quotes, key=lambda q: q.timestamp)
    
    # Create retrieved lookup again
    retrieved_lookup = {q.timestamp.replace(microsecond=0): q for q in sorted_retrieved}
    
    # Print first 20 and last 20 for inspection (to avoid too much output)
    print("\n--- FIRST 20 QUOTES ---")
    for i, orig_quote in enumerate(sorted_original[:20]):
        normalized_ts = orig_quote.timestamp.replace(microsecond=0)
        date_str = orig_quote.timestamp.strftime('%Y-%m-%d')
        time_str = orig_quote.timestamp.strftime('%H:%M:%S')
        
        if normalized_ts in retrieved_lookup:
            retr_quote = retrieved_lookup[normalized_ts]
            # Print each field
            for field in ['open', 'high', 'low', 'close', 'volume']:
                orig_val = getattr(orig_quote, field)
                retr_val = getattr(retr_quote, field)
                match = "OK" if abs(orig_val - retr_val) < 0.01 else "FAIL"
                print(f"{date_str:<12} {time_str:<10} {field:<8} {orig_val:<15.2f} {retr_val:<15.2f} {match:<8}")
        else:
            print(f"{date_str:<12} {time_str:<10} {'MISSING':<8} {'N/A':<15} {'N/A':<15} {'FAIL':<8}")
    
    if len(sorted_original) > 40:
        print(f"\n... ({len(sorted_original) - 40} quotes omitted) ...\n")
        print("--- LAST 20 QUOTES ---")
        for orig_quote in sorted_original[-20:]:
            normalized_ts = orig_quote.timestamp.replace(microsecond=0)
            date_str = orig_quote.timestamp.strftime('%Y-%m-%d')
            time_str = orig_quote.timestamp.strftime('%H:%M:%S')
            
            if normalized_ts in retrieved_lookup:
                retr_quote = retrieved_lookup[normalized_ts]
                # Print each field
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    orig_val = getattr(orig_quote, field)
                    retr_val = getattr(retr_quote, field)
                    match = "OK" if abs(orig_val - retr_val) < 0.01 else "FAIL"
                    print(f"{date_str:<12} {time_str:<10} {field:<8} {orig_val:<15.2f} {retr_val:<15.2f} {match:<8}")
            else:
                print(f"{date_str:<12} {time_str:<10} {'MISSING':<8} {'N/A':<15} {'N/A':<15} {'FAIL':<8}")
    elif len(sorted_original) > 20:
        print("\n--- REMAINING QUOTES ---")
        for orig_quote in sorted_original[20:]:
            normalized_ts = orig_quote.timestamp.replace(microsecond=0)
            date_str = orig_quote.timestamp.strftime('%Y-%m-%d')
            time_str = orig_quote.timestamp.strftime('%H:%M:%S')
            
            if normalized_ts in retrieved_lookup:
                retr_quote = retrieved_lookup[normalized_ts]
                # Print each field
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    orig_val = getattr(orig_quote, field)
                    retr_val = getattr(retr_quote, field)
                    match = "OK" if abs(orig_val - retr_val) < 0.01 else "FAIL"
                    print(f"{date_str:<12} {time_str:<10} {field:<8} {orig_val:<15.2f} {retr_val:<15.2f} {match:<8}")
            else:
                print(f"{date_str:<12} {time_str:<10} {'MISSING':<8} {'N/A':<15} {'N/A':<15} {'FAIL':<8}")
    
    print("="*120)
    
    # Print summary of value matching
    if value_mismatches:
        print(f"\n[WARNING] Found {len(value_mismatches)} value mismatches:")
        for mismatch in value_mismatches[:10]:  # Show first 10
            print(f"  - {mismatch['timestamp']}: {mismatch['issue']}")
            if 'mismatched_fields' in mismatch:
                for field_mismatch in mismatch['mismatched_fields']:
                    print(f"    {field_mismatch['field']}: {field_mismatch['original']} vs {field_mismatch['retrieved']}")
    else:
        print("\n[SUCCESS] All values match perfectly!")


if __name__ == "__main__":
    asyncio.run(main())

