"""
Script to clear LLM trend detection cache files.

This will delete all cached LLM trend states from the file system.
"""

import sys
from pathlib import Path
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def clear_llm_cache():
    """Clear all LLM trend cache files."""
    # Check multiple possible locations
    possible_dirs = [
        Path("data_cache/llm_trend"),
        Path("scripts/data_cache/llm_trend"),
        project_root / "data_cache" / "llm_trend",
        project_root / "scripts" / "data_cache" / "llm_trend",
    ]
    
    print("=" * 60)
    print("LLM Trend Cache Cleanup")
    print("=" * 60)
    
    all_cache_files = []
    for cache_dir in possible_dirs:
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.json.tmp"))
            all_cache_files.extend(cache_files)
            if cache_files:
                print(f"\nFound cache directory: {cache_dir}")
    
    if not all_cache_files:
        print(f"\n✓ No cache files found")
        print("  Cache is already empty.")
        return
    
    print(f"\nFound {len(all_cache_files)} cache file(s):")
    for cache_file in all_cache_files:
        print(f"  - {cache_file}")
    
    # Ask for confirmation (but since this is a script, we'll just delete)
    print(f"\nDeleting cache files...")
    
    deleted_count = 0
    for cache_file in all_cache_files:
        try:
            cache_file.unlink()
            deleted_count += 1
            print(f"  ✓ Deleted: {cache_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to delete {cache_file.name}: {e}")
    
    # Also clear in-memory cache if possible
    try:
        from core.models.llm_trend import _cache_instance, _registry
        # Clear the global registry
        _registry.clear()
        print(f"✓ Cleared in-memory registry")
        
        # Clear cache instance if it exists
        if _cache_instance is not None:
            _cache_instance._in_memory_cache.clear()
            print(f"✓ Cleared cache instance in-memory cache")
    except Exception as e:
        print(f"  Note: Could not clear in-memory cache: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Cache cleared: {deleted_count} file(s) deleted")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        clear_llm_cache()
    except KeyboardInterrupt:
        print("\n\nCache cleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error clearing cache: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

