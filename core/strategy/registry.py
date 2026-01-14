"""
Strategy Registry for dynamic strategy discovery and import.

This module provides utilities to discover and import Strategy classes
from the core/strategy/ directory, making it easy to add new strategies
without hardcoding imports.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Type, Optional, Tuple

from core.strategy.base import Strategy

logger = logging.getLogger(__name__)

# Strategy registry cache
_STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}
_CONFIG_REGISTRY: Dict[str, Type] = {}


def get_project_root() -> Path:
    """
    Get project root directory using multiple fallback methods.
    
    Returns:
        Path to project root
    """
    # Method 1: Check if /app exists (Railway/Docker)
    if Path('/app').exists() and (Path('/app') / 'core' / 'strategy').exists():
        return Path('/app')
    
    # Method 2: Calculate from this file's location
    current = Path(__file__).resolve()
    # From core/strategy/registry.py -> core/strategy -> core -> project root
    project_root = current.parent.parent.parent
    
    if (project_root / 'core' / 'strategy').exists():
        return project_root
    
    # Method 3: Try current working directory
    cwd = Path.cwd()
    if (cwd / 'core' / 'strategy').exists():
        return cwd
    
    # Fallback: return calculated path anyway
    return project_root


def _discover_strategies(project_root: Optional[Path] = None) -> Tuple[Dict[str, Type[Strategy]], Dict[str, Type]]:
    """
    Dynamically discover all Strategy classes in core/strategy/.
    
    Args:
        project_root: Optional project root path. If None, will be auto-detected.
        
    Returns:
        Tuple of (strategy_registry, config_registry) dictionaries
    """
    if project_root is None:
        project_root = get_project_root()
    
    strategy_dir = project_root / "core" / "strategy"
    
    if not strategy_dir.exists():
        logger.warning(f"Strategy directory not found: {strategy_dir}")
        return {}, {}
    
    strategies = {}
    configs = {}
    
    # Files to exclude
    exclude = {
        "base",           # Base class
        "strategy_utils", # Utility functions
        "contributions",  # Contribution manager
        "example_strategy", # Example/template
        "__pycache__",    # Python cache
        "old",            # Old strategies directory
        "__init__",       # Init file
    }
    
    for file in strategy_dir.glob("*.py"):
        if file.stem in exclude:
            continue
        
        try:
            module_name = f"core.strategy.{file.stem}"
            module = importlib.import_module(module_name)
            
            # Find all Strategy subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a Strategy subclass (but not Strategy itself)
                if (issubclass(obj, Strategy) and 
                    obj is not Strategy and 
                    obj.__module__ == module_name):
                    strategies[name] = obj
                    logger.debug(f"Discovered strategy: {name} from {module_name}")
                
                # Also look for Config classes (usually named *Config)
                if (name.endswith("Config") and 
                    obj.__module__ == module_name and
                    not name.startswith("_")):
                    configs[name] = obj
                    logger.debug(f"Discovered config: {name} from {module_name}")
                    
        except Exception as e:
            # Skip modules that can't be imported (e.g., missing dependencies)
            logger.debug(f"Could not import strategy from {file.stem}: {e}")
            continue
    
    return strategies, configs


def get_strategy_class(strategy_name: str, project_root: Optional[Path] = None) -> Optional[Type[Strategy]]:
    """
    Get a strategy class by name.
    
    Args:
        strategy_name: Name of the strategy class (e.g., "LLMTrendDetectionStrategy")
        project_root: Optional project root path
        
    Returns:
        Strategy class or None if not found
    """
    if not _STRATEGY_REGISTRY:
        strategies, configs = _discover_strategies(project_root)
        _STRATEGY_REGISTRY.update(strategies)
        _CONFIG_REGISTRY.update(configs)
    
    return _STRATEGY_REGISTRY.get(strategy_name)


def get_config_class(config_name: str, project_root: Optional[Path] = None) -> Optional[Type]:
    """
    Get a config class by name.
    
    Args:
        config_name: Name of the config class (e.g., "LLMTrendDetectionConfig")
        project_root: Optional project root path
        
    Returns:
        Config class or None if not found
    """
    if not _CONFIG_REGISTRY:
        strategies, configs = _discover_strategies(project_root)
        _STRATEGY_REGISTRY.update(strategies)
        _CONFIG_REGISTRY.update(configs)
    
    return _CONFIG_REGISTRY.get(config_name)


def list_strategies(project_root: Optional[Path] = None) -> Dict[str, Type[Strategy]]:
    """
    List all available strategies.
    
    Args:
        project_root: Optional project root path
        
    Returns:
        Dictionary mapping strategy name to Strategy class
    """
    if not _STRATEGY_REGISTRY:
        strategies, configs = _discover_strategies(project_root)
        _STRATEGY_REGISTRY.update(strategies)
        _CONFIG_REGISTRY.update(configs)
    
    return _STRATEGY_REGISTRY.copy()


def list_configs(project_root: Optional[Path] = None) -> Dict[str, Type]:
    """
    List all available config classes.
    
    Args:
        project_root: Optional project root path
        
    Returns:
        Dictionary mapping config name to Config class
    """
    if not _CONFIG_REGISTRY:
        strategies, configs = _discover_strategies(project_root)
        _STRATEGY_REGISTRY.update(strategies)
        _CONFIG_REGISTRY.update(configs)
    
    return _CONFIG_REGISTRY.copy()


def clear_cache():
    """Clear the strategy and config registry cache (useful for testing)."""
    global _STRATEGY_REGISTRY, _CONFIG_REGISTRY
    _STRATEGY_REGISTRY.clear()
    _CONFIG_REGISTRY.clear()
