from __future__ import annotations

import json
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class LLMTrendState:
    symbol: str
    timeframe: str
    as_of: datetime

    regime_final: str
    trend_short: str
    trend_medium: str
    trend_long: str

    trend_strength: float
    range_strength: float

    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    explanation: str = ""
    summary_for_user: str = ""

    # Raw JSON from LLM for downstream agents / caching
    raw_result: Dict[str, Any] = field(default_factory=dict)
    
    # Flag to indicate if this result came from LLM (True) or fallback (False)
    # Only LLM results should be cached
    from_llm: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data['as_of'] = self.as_of.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMTrendState':
        """Create from dictionary (from JSON deserialization)."""
        # Convert ISO format string back to datetime
        if isinstance(data.get('as_of'), str):
            data['as_of'] = datetime.fromisoformat(data['as_of'])
        # Handle backward compatibility: old cache entries without 'from_llm' default to True
        if 'from_llm' not in data:
            data['from_llm'] = True
        return cls(**data)


class LLMTrendCache:
    """
    Manages file-based caching of LLM trend states using JSON format.
    
    Follows the same pattern as DataCache for consistency:
    - File-based persistence
    - In-memory cache for fast access
    - Same normalization logic
    - Atomic writes
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the LLM trend cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses project root / "data_cache/llm_trend"
        """
        if cache_dir is None:
            # Use absolute path relative to project root to avoid issues with script working directory
            import os
            # Try to find project root by looking for common markers
            current = Path.cwd()
            project_root = current
            # Look for project root markers (like .git, pyproject.toml, etc.)
            while project_root != project_root.parent:
                if (project_root / ".git").exists() or (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
                    break
                project_root = project_root.parent
            cache_dir = str(project_root / "data_cache" / "llm_trend")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache: (symbol, timeframe, mtime) -> Dict[date, LLMTrendState]
        self._in_memory_cache: Dict[Tuple[str, str, Optional[float]], Dict[date, LLMTrendState]] = {}
        logger.info(f"LLMTrendCache initialized with cache directory: {self._cache_dir}")
    
    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        """
        Normalize timeframe to a canonical format for consistent cache file naming.
        
        Uses the same logic as DataCache for consistency.
        
        Args:
            timeframe: Timeframe string in any format
            
        Returns:
            Normalized timeframe string
        """
        if not timeframe:
            raise ValueError("Timeframe cannot be empty")
        
        timeframe_upper = timeframe.upper()
        
        # Handle single letter timeframes (D, H, W, M, Y)
        if timeframe_upper == "D":
            return "1D"
        elif timeframe_upper == "H":
            return "1H"
        elif timeframe_upper == "W":
            return "1W"
        elif timeframe_upper == "M":
            return "1M"
        elif timeframe_upper == "Y":
            return "1Y"
        
        # Handle minutely vs monthly distinction
        if timeframe_upper.endswith("M"):
            prefix = timeframe_upper[:-1]
            if not prefix or (len(prefix) == 1 and prefix.isdigit()):
                return (prefix if prefix else "1") + "M"
            elif prefix.isdigit():
                return prefix + "m"
        elif timeframe.endswith("m"):
            prefix = timeframe[:-1]
            if prefix.isdigit():
                return prefix + "m"
        
        # Handle pure numeric minutely resolutions
        if timeframe.isdigit():
            return timeframe + "m"
        
        # Handle multi-period resolutions
        if any(timeframe_upper.endswith(suffix) for suffix in ["H", "D", "W", "M", "Y"]):
            for suffix in ["H", "D", "W", "M", "Y"]:
                if timeframe_upper.endswith(suffix):
                    prefix = timeframe_upper[:-len(suffix)]
                    if prefix.isdigit() or prefix == "":
                        return (prefix if prefix else "1") + suffix
        
        return timeframe_upper
    
    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get the cache file path for a symbol and timeframe.
        
        Uses normalized timeframe to ensure consistent cache file naming.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timeframe: Timeframe in any format (e.g., "D", "1D", "15m")
            
        Returns:
            Path to the cache file
        """
        normalized_symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        cache_filename = f"{normalized_symbol}_{normalized_timeframe}.json"
        return self._cache_dir / cache_filename
    
    def _get_all_cached_states(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[date, LLMTrendState]]:
        """
        Get all cached states from in-memory cache or file.
        
        Similar to DataCache.get_all_cached_bars().
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary mapping date -> LLMTrendState, or None if not available
        """
        normalized_symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        # First, try to find any cache entry for this symbol/timeframe without file I/O
        best_match = None
        best_mtime = 0
        for key, states_dict in self._in_memory_cache.items():
            if key[0] == normalized_symbol and key[1] == normalized_timeframe:
                key_mtime = key[2] if key[2] is not None else 0
                if key_mtime >= best_mtime:
                    best_mtime = key_mtime
                    best_match = states_dict
        
        # If we found a match in memory, return it immediately
        if best_match is not None:
            return best_match
        
        # Not in memory - check file system and load if exists
        cache_path = self.get_cache_path(symbol, timeframe)
        if not cache_path.exists():
            return None
        
        # Load from file
        try:
            cache_mtime = cache_path.stat().st_mtime
            cache_key = (normalized_symbol, normalized_timeframe, cache_mtime)
            
            # Check if already in memory with this exact key
            if cache_key in self._in_memory_cache:
                return self._in_memory_cache[cache_key]
            
            # Load from file
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Convert to Dict[date, LLMTrendState]
            states_dict: Dict[date, LLMTrendState] = {}
            for state_dict in data.get('states', []):
                try:
                    state = LLMTrendState.from_dict(state_dict)
                    state_date = state.as_of.date()
                    states_dict[state_date] = state
                except Exception as e:
                    logger.warning(f"Failed to load state from cache file {cache_path}: {e}")
                    continue
            
            if not states_dict:
                return None
            
            # Store in memory for future requests
            self._in_memory_cache[cache_key] = states_dict
            
            # Remove any older cache entries for this symbol/timeframe
            keys_to_remove = [
                k for k in self._in_memory_cache.keys()
                if k[0] == normalized_symbol and k[1] == normalized_timeframe and k != cache_key
            ]
            for key in keys_to_remove:
                del self._in_memory_cache[key]
            
            return states_dict
            
        except Exception as e:
            logger.warning(
                f"Failed to load cache from {cache_path}: {e}. "
                "Will fetch from LLM instead."
            )
            # If cache is corrupted, delete it
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None
    
    def get_state(
        self,
        symbol: str,
        timeframe: str,
        as_of_date: Optional[date] = None,
    ) -> Optional[LLMTrendState]:
        """
        Get state from cache.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            as_of_date: Specific date to look up. If None, returns the latest state.
            
        Returns:
            LLMTrendState if found, None otherwise
        """
        all_states = self._get_all_cached_states(symbol, timeframe)
        if not all_states:
            return None
        
        if as_of_date:
            # Look up specific date
            return all_states.get(as_of_date)
        else:
            # Return latest state
            if not all_states:
                return None
            latest_date = max(all_states.keys())
            return all_states[latest_date]
    
    async def save_state(self, state: LLMTrendState) -> None:
        """
        Save state to cache, merging with existing data if present.
        
        Similar to DataCache.save_bars().
        
        Args:
            state: LLMTrendState to save
        """
        symbol = state.symbol
        timeframe = state.timeframe
        state_date = state.as_of.date()
        
        cache_path = self.get_cache_path(symbol, timeframe)
        
        try:
            # Load existing cache if it exists
            existing_states = self._get_all_cached_states(symbol, timeframe)
            if existing_states is None:
                existing_states = {}
            
            # Add or update state
            existing_states[state_date] = state
            
            # Convert to list of dicts for JSON
            states_list = [s.to_dict() for s in existing_states.values()]
            
            # Write to temporary file first (atomic write)
            temp_path = cache_path.with_suffix('.json.tmp')
            
            # Remove temp file if it exists from a previous failed write
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            
            # Write to temp file
            def _write():
                with open(temp_path, 'w') as f:
                    json.dump({'states': states_list}, f, indent=2)
            
            await asyncio.to_thread(_write)
            
            # Atomic rename
            temp_path.replace(cache_path)
            
            # Update in-memory cache
            normalized_symbol = symbol.upper()
            normalized_timeframe = self._normalize_timeframe(timeframe)
            try:
                cache_mtime = cache_path.stat().st_mtime
                cache_key = (normalized_symbol, normalized_timeframe, cache_mtime)
            except Exception:
                cache_key = (normalized_symbol, normalized_timeframe, None)
            
            self._in_memory_cache[cache_key] = existing_states
            
            # Remove any older cache entries for this symbol/timeframe
            keys_to_remove = [
                k for k in self._in_memory_cache.keys()
                if k[0] == normalized_symbol and k[1] == normalized_timeframe and k != cache_key
            ]
            for key in keys_to_remove:
                del self._in_memory_cache[key]
            
            logger.debug(f"Saved LLM trend state for {symbol} ({timeframe}) on {state_date}")
            
        except Exception as e:
            logger.warning(f"Failed to save LLM trend state to cache: {e}")
    
    def get_all_states(self) -> Dict[Tuple[str, str, date], LLMTrendState]:
        """
        Get all states from in-memory cache.
        
        Returns:
            Dictionary mapping (symbol, timeframe, date) -> LLMTrendState
        """
        result: Dict[Tuple[str, str, date], LLMTrendState] = {}
        for (symbol, timeframe, _), states_dict in self._in_memory_cache.items():
            for state_date, state in states_dict.items():
                result[(symbol, timeframe, state_date)] = state
        return result


# Global cache instance
_cache_instance: Optional[LLMTrendCache] = None


def _get_cache() -> LLMTrendCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMTrendCache()
    return _cache_instance


# Public API functions (backward compatible)
async def update_state(state: LLMTrendState) -> None:
    """Update state in cache (async)."""
    cache = _get_cache()
    await cache.save_state(state)


def get_state(symbol: str, timeframe: str, as_of_date: Optional[date] = None) -> Optional[LLMTrendState]:
    """Get state from cache."""
    cache = _get_cache()
    return cache.get_state(symbol, timeframe, as_of_date)


def get_all_states() -> Dict[Tuple[str, str, date], LLMTrendState]:
    """Get all states from cache."""
    cache = _get_cache()
    return cache.get_all_states()
