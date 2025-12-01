"""
Data Engine Factory - Creates data engines based on configuration.

This module provides a factory function to create data engines from configuration,
supporting multiple data sources (marketdata.app, yfinance, etc.) with a unified interface.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging

from core.data.base import DataEngine

logger = logging.getLogger(__name__)

# Lazy import for cached engine (has optional dependencies)
try:
    from core.data.cached_engine import CachedDataEngine
    _CACHED_ENGINE_AVAILABLE = True
except ImportError:
    CachedDataEngine = None  # type: ignore
    _CACHED_ENGINE_AVAILABLE = False

# Lazy imports to avoid dependency issues
_MARKETDATA_AVAILABLE = False
_YFINANCE_AVAILABLE = False

try:
    from core.data.marketdata_app import MarketDataAppAdapter
    _MARKETDATA_AVAILABLE = True
except ImportError:
    MarketDataAppAdapter = None  # type: ignore

try:
    from core.data.yfinance_data import YFinanceDataAdapter
    _YFINANCE_AVAILABLE = True
except ImportError:
    YFinanceDataAdapter = None  # type: ignore


def create_data_engine(
    source_name: str,
    config: Dict[str, Any],
    cache_enabled: bool = True,
    cache_dir: str = "data_cache/bars",
) -> DataEngine:
    """
    Create a data engine based on source name and configuration.
    
    Args:
        source_name: Name of the data source ("marketdata_app", "yfinance", etc.)
        config: Configuration dictionary for the data source
        cache_enabled: Whether to enable caching (default: True)
        cache_dir: Directory for cache files (default: "data_cache/bars")
    
    Returns:
        DataEngine instance (wrapped with CachedDataEngine if caching enabled)
    
    Raises:
        ValueError: If source_name is not supported or required config is missing
        ImportError: If required dependencies are not available
    """
    source_name_lower = source_name.lower()
    
    # Create base engine based on source
    base_engine: Optional[DataEngine] = None
    
    if source_name_lower == "marketdata_app" or source_name_lower == "marketdata":
        if not _MARKETDATA_AVAILABLE:
            raise ImportError(
                "MarketDataAppAdapter is not available. "
                "Make sure core.data.marketdata_app can be imported."
            )
        
        if "api_token" not in config:
            raise ValueError(
                f"Missing 'api_token' in config for data source '{source_name}'. "
                f"Please set 'data_sources.{source_name}.api_token' in your config file."
            )
        
        timeout = config.get("timeout", 30.0)
        max_retries = config.get("max_retries", 3)
        
        base_engine = MarketDataAppAdapter(
            api_token=config["api_token"],
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info(f"Created MarketDataAppAdapter data engine")
        
    elif source_name_lower == "yfinance":
        if not _YFINANCE_AVAILABLE:
            raise ImportError(
                "YFinanceDataAdapter is not available. "
                "Make sure yfinance is installed: pip install yfinance"
            )
        
        # yfinance doesn't require any config (no API key needed)
        base_engine = YFinanceDataAdapter()
        logger.info(f"Created YFinanceDataAdapter data engine")
        
    else:
        raise ValueError(
            f"Unsupported data source: '{source_name}'. "
            f"Supported sources: 'marketdata_app', 'yfinance'"
        )
    
    if base_engine is None:
        raise RuntimeError(f"Failed to create data engine for source '{source_name}'")
    
    # Wrap with caching if enabled
    if cache_enabled:
        if not _CACHED_ENGINE_AVAILABLE:
            logger.warning(
                "Caching requested but CachedDataEngine is not available. "
                "Returning base engine without caching."
            )
            return base_engine
        
        cached_engine = CachedDataEngine(
            base_engine=base_engine,
            cache_dir=cache_dir,
            cache_enabled=True,
        )
        logger.info(f"Wrapped data engine with caching (cache_dir: {cache_dir})")
        return cached_engine
    else:
        return base_engine


def create_data_engine_from_config(
    env_config: Dict[str, Any],
    use_for: str = "historical",  # "historical" or "realtime"
    cache_enabled: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> DataEngine:
    """
    Create a data engine from the full environment config.
    
    Expected config structure:
        data:
          historical_source: "marketdata_app"  # or "yfinance"
          realtime_source: "marketdata_app"    # or "yfinance", "moomoo"
          cache_enabled: true
          cache_dir: "data_cache/bars"
        
        data_sources:
          marketdata_app:
            enabled: true
            api_token: "..."
            timeout: 30.0
            max_retries: 3
          yfinance:
            enabled: true
    
    Args:
        env_config: The full environment config dictionary
        use_for: Whether to use "historical" or "realtime" source
        cache_enabled: Whether to enable caching (overrides config if provided)
        cache_dir: Directory for cache files (overrides config if provided)
    
    Returns:
        DataEngine instance
    
    Raises:
        ValueError: If config is invalid or missing required fields
    """
    # Get data section (new structure)
    data_config = env_config.get("data", {})
    
    # Get data_sources section
    data_sources_config = env_config.get("data_sources", {})
    
    # Backward compatibility: if old "marketdata" structure exists, migrate it
    if "marketdata" in env_config and "data_sources" not in env_config:
        logger.warning(
            "Using legacy 'marketdata' config structure. "
            "Please migrate to new 'data_sources' structure. "
            "See config/env.backtest.yaml for the new format."
        )
        # Migrate old structure to new
        data_sources_config = {
            "marketdata_app": {
                "enabled": True,
                "api_token": env_config["marketdata"].get("api_token", ""),
            }
        }
        data_config = {
            "historical_source": "marketdata_app",
            "realtime_source": "marketdata_app",
            "cache_enabled": True,
            "cache_dir": "data_cache/bars",
        }
    
    # Determine which source to use
    if use_for == "historical":
        source_name = data_config.get("historical_source", "marketdata_app")
    elif use_for == "realtime":
        source_name = data_config.get("realtime_source", "marketdata_app")
    else:
        raise ValueError(f"Invalid 'use_for' parameter: {use_for}. Must be 'historical' or 'realtime'")
    
    # Get source-specific config
    source_config = get_data_source_config(data_sources_config, source_name)
    
    # Get cache settings
    use_cache = cache_enabled if cache_enabled is not None else data_config.get("cache_enabled", True)
    cache_directory = cache_dir if cache_dir else data_config.get("cache_dir", "data_cache/bars")
    
    # Create the data engine
    engine = create_data_engine(
        source_name=source_name,
        config=source_config,
        cache_enabled=use_cache,
        cache_dir=cache_directory,
    )
    
    logger.info(f"Created data engine: {source_name} (for {use_for} data, caching: {use_cache})")
    return engine


def get_data_source_config(
    data_sources_config: Dict[str, Any],
    source_name: str,
) -> Dict[str, Any]:
    """
    Get configuration for a specific data source.
    
    Args:
        data_sources_config: The 'data_sources' section from config file
        source_name: Name of the data source to get config for
    
    Returns:
        Configuration dictionary for the data source
    
    Raises:
        ValueError: If source is not found or not enabled
    """
    source_name_lower = source_name.lower()
    
    # Try exact match first
    if source_name_lower in data_sources_config:
        source_config = data_sources_config[source_name_lower]
    elif source_name_lower == "marketdata_app" and "marketdata" in data_sources_config:
        # Backward compatibility: "marketdata" -> "marketdata_app"
        source_config = data_sources_config["marketdata"]
    else:
        raise ValueError(
            f"Data source '{source_name}' not found in config. "
            f"Available sources: {list(data_sources_config.keys())}"
        )
    
    # Check if source is enabled
    if isinstance(source_config, dict) and source_config.get("enabled", True) is False:
        raise ValueError(f"Data source '{source_name}' is disabled in config")
    
    # If config is just a string (e.g., api_token as direct value), wrap it
    if not isinstance(source_config, dict):
        # Backward compatibility: if it's just the api_token string
        if source_name_lower in ("marketdata_app", "marketdata"):
            return {"api_token": source_config}
        else:
            return {}
    
    return source_config

