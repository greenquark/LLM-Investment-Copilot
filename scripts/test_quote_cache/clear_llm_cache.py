"""
Script to clear LLM trend detection cache files.

This script requires a parameter to specify what to clear:
- Clear all cache files: use --all or --erase-all
- Clear cache entries for specific dates: use --date, --start-date, and/or --end-date

Usage:
    # Clear all cache files (REQUIRED: must specify --all)
    python clear_llm_cache.py --all

    # Clear cache for a specific date
    python clear_llm_cache.py --date 2024-01-15

    # Clear cache for a date range
    python clear_llm_cache.py --start-date 2024-01-01 --end-date 2024-01-31

    # Clear cache from a date onwards
    python clear_llm_cache.py --start-date 2024-01-15

    # Clear cache up to a date
    python clear_llm_cache.py --end-date 2024-01-15

    # Dry run (show what would be deleted)
    python clear_llm_cache.py --all --dry-run
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_cache_directories() -> List[Path]:
    """Find all possible cache directories."""
    possible_dirs = [
        Path("data_cache/llm_trend"),
        Path("scripts/data_cache/llm_trend"),
        project_root / "data_cache" / "llm_trend",
        project_root / "scripts" / "data_cache" / "llm_trend",
    ]
    
    found_dirs = []
    for cache_dir in possible_dirs:
        if cache_dir.exists():
            found_dirs.append(cache_dir)
    
    return found_dirs


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def should_remove_state(state_date: date, start_date: Optional[date], end_date: Optional[date]) -> bool:
    """Check if a state should be removed based on date criteria."""
    if start_date is None and end_date is None:
        # No date filter - remove all
        return True
    
    if start_date is not None and end_date is not None:
        # Date range
        return start_date <= state_date <= end_date
    elif start_date is not None:
        # Start date only (remove from start_date onwards)
        return state_date >= start_date
    elif end_date is not None:
        # End date only (remove up to end_date)
        return state_date <= end_date
    
    return False


def clear_llm_cache(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    dry_run: bool = False
):
    """
    Clear LLM trend cache files or entries.
    
    Args:
        start_date: Start date for date range (inclusive). If None and end_date is set, clears up to end_date.
        end_date: End date for date range (inclusive). If None and start_date is set, clears from start_date onwards.
        dry_run: If True, only show what would be deleted without actually deleting.
    """
    cache_dirs = find_cache_directories()
    
    print("=" * 60)
    print("LLM Trend Cache Cleanup")
    print("=" * 60)
    
    if start_date or end_date:
        if start_date and end_date:
            print(f"\nDate range: {start_date} to {end_date}")
        elif start_date:
            print(f"\nFrom date: {start_date} onwards")
        elif end_date:
            print(f"\nUp to date: {end_date}")
    else:
        print("\nMode: Clear all cache files")
    
    if dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")
    
    if not cache_dirs:
        print("\n[OK] No cache directories found")
        return
    
    all_cache_files = []
    for cache_dir in cache_dirs:
        cache_files = list(cache_dir.glob("*.json"))
        # Exclude .tmp files
        cache_files = [f for f in cache_files if not f.name.endswith('.tmp')]
        all_cache_files.extend(cache_files)
        if cache_files:
            print(f"\nFound cache directory: {cache_dir}")
    
    if not all_cache_files:
        print(f"\n[OK] No cache files found")
        print("  Cache is already empty.")
        return
    
    print(f"\nFound {len(all_cache_files)} cache file(s)")
    
    # If no date filter, delete entire files
    if start_date is None and end_date is None:
        print(f"\nDeleting cache files...")
        deleted_count = 0
        for cache_file in all_cache_files:
            try:
                if not dry_run:
                    cache_file.unlink()
                deleted_count += 1
                print(f"  [OK] {'Would delete' if dry_run else 'Deleted'}: {cache_file.name}")
            except Exception as e:
                print(f"  [X] Failed to delete {cache_file.name}: {e}")
        
        # Also clear in-memory cache if possible
        if not dry_run:
            try:
                from core.models.llm_trend import _cache_instance, _registry
                _registry.clear()
                print(f"[OK] Cleared in-memory registry")
                
                if _cache_instance is not None:
                    _cache_instance._in_memory_cache.clear()
                    print(f"[OK] Cleared cache instance in-memory cache")
            except Exception as e:
                print(f"  Note: Could not clear in-memory cache: {e}")
        
        print(f"\n{'=' * 60}")
        print(f"[OK] {'Would clear' if dry_run else 'Cleared'} cache: {deleted_count} file(s) {'would be deleted' if dry_run else 'deleted'}")
        print(f"{'=' * 60}")
        return
    
    # Date range filtering - modify cache files
    print(f"\nFiltering cache entries by date...")
    
    total_states_removed = 0
    files_modified = 0
    files_deleted = 0
    
    for cache_file in all_cache_files:
        try:
            # Load cache file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            states = cache_data.get('states', [])
            if not states:
                continue
            
            # Filter states by date
            original_count = len(states)
            filtered_states = []
            removed_count = 0
            
            for state_dict in states:
                try:
                    # Parse the as_of date
                    as_of_str = state_dict.get('as_of')
                    if isinstance(as_of_str, str):
                        state_datetime = datetime.fromisoformat(as_of_str.replace('Z', '+00:00'))
                        state_date = state_datetime.date()
                    else:
                        # Skip if we can't parse the date
                        filtered_states.append(state_dict)
                        continue
                    
                    # Check if this state should be removed
                    if should_remove_state(state_date, start_date, end_date):
                        removed_count += 1
                    else:
                        filtered_states.append(state_dict)
                except Exception as e:
                    # If we can't parse a state, keep it
                    print(f"  Warning: Could not parse state in {cache_file.name}: {e}")
                    filtered_states.append(state_dict)
            
            if removed_count == 0:
                print(f"  - {cache_file.name}: No matching entries (kept {original_count} states)")
                continue
            
            if len(filtered_states) == 0:
                # All states removed - delete the file
                if not dry_run:
                    cache_file.unlink()
                files_deleted += 1
                total_states_removed += removed_count
                print(f"  [OK] {'Would delete' if dry_run else 'Deleted'}: {cache_file.name} (all {removed_count} states removed)")
            else:
                # Update the file with remaining states
                if not dry_run:
                    cache_data['states'] = filtered_states
                    # Write to temporary file first (atomic write)
                    temp_path = cache_file.with_suffix('.json.tmp')
                    with open(temp_path, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    # Atomic rename
                    temp_path.replace(cache_file)
                
                files_modified += 1
                total_states_removed += removed_count
                print(f"  [OK] {'Would update' if dry_run else 'Updated'}: {cache_file.name} (removed {removed_count}/{original_count} states, kept {len(filtered_states)})")
        
        except Exception as e:
            print(f"  [X] Failed to process {cache_file.name}: {e}")
    
    # Clear in-memory cache if we modified files
    if not dry_run and (files_modified > 0 or files_deleted > 0):
        try:
            from core.models.llm_trend import _cache_instance, _registry
            _registry.clear()
            print(f"\n[OK] Cleared in-memory registry")
            
            if _cache_instance is not None:
                _cache_instance._in_memory_cache.clear()
                print(f"[OK] Cleared cache instance in-memory cache")
        except Exception as e:
            print(f"  Note: Could not clear in-memory cache: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"[OK] {'Would clear' if dry_run else 'Cleared'} cache:")
    print(f"  - {total_states_removed} state(s) {'would be removed' if dry_run else 'removed'}")
    print(f"  - {files_modified} file(s) {'would be modified' if dry_run else 'modified'}")
    print(f"  - {files_deleted} file(s) {'would be deleted' if dry_run else 'deleted'}")
    print(f"{'=' * 60}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Clear LLM trend detection cache files or entries by date. A parameter is REQUIRED.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear all cache files (REQUIRED: must specify --all)
  python clear_llm_cache.py --all

  # Clear cache for a specific date
  python clear_llm_cache.py --date 2024-01-15

  # Clear cache for a date range
  python clear_llm_cache.py --start-date 2024-01-01 --end-date 2024-01-31

  # Clear cache from a date onwards
  python clear_llm_cache.py --start-date 2024-01-15

  # Clear cache up to a date
  python clear_llm_cache.py --end-date 2024-01-15

  # Dry run (show what would be deleted)
  python clear_llm_cache.py --all --dry-run
        """
    )
    
    parser.add_argument(
        '--all',
        '--erase-all',
        dest='erase_all',
        action='store_true',
        help='Clear all cache files. This is required if no date parameters are provided.'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Clear cache for a specific date (YYYY-MM-DD). Equivalent to --start-date and --end-date with the same value.'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for date range (YYYY-MM-DD, inclusive). If set without --end-date, clears from this date onwards.'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for date range (YYYY-MM-DD, inclusive). If set without --start-date, clears up to this date.'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting anything.'
    )
    
    args = parser.parse_args()
    
    # Validate that at least one parameter is provided
    has_date_param = bool(args.date or args.start_date or args.end_date)
    if not args.erase_all and not has_date_param:
        parser.error("At least one parameter is required. Use --all to clear all cache, or provide date parameters (--date, --start-date, --end-date).")
    
    # Validate that --all is not used with date parameters
    if args.erase_all and has_date_param:
        parser.error("Cannot use --all with date parameters. Use --all alone to clear all cache, or use date parameters without --all.")
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.erase_all:
        # Clear all - no date filtering
        start_date = None
        end_date = None
    elif args.date:
        # Single date - set both start and end to the same date
        parsed_date = parse_date(args.date)
        start_date = parsed_date
        end_date = parsed_date
    else:
        if args.start_date:
            start_date = parse_date(args.start_date)
        if args.end_date:
            end_date = parse_date(args.end_date)
    
    # Validate date range
    if start_date and end_date and start_date > end_date:
        print(f"Error: Start date ({start_date}) is after end date ({end_date})")
        sys.exit(1)
    
    try:
        clear_llm_cache(start_date=start_date, end_date=end_date, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nCache cleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Error clearing cache: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

