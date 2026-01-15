"""
Centralized LLM configuration loader.

This module provides utilities to load LLM model configuration from
config/env.backtest.yaml, providing a single source of truth for
model selection and fallback behavior.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from core.utils.config_loader import load_config_with_secrets


def load_llm_config(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load centralized LLM configuration from config/env.backtest.yaml.
    
    Args:
        project_root: Path to project root directory. If None, attempts to find it.
    
    Returns:
        Dictionary containing LLM configuration with keys:
        - primary_model: str
        - fallback_models: List[str]
        - temperature: float
        - max_tokens: int
        - max_completion_tokens: int
        - timeout: float
        - model_capabilities: Dict[str, Dict[str, Any]]
    """
    if project_root is None:
        # Try to find project root by looking for common markers
        current = Path(__file__).resolve()
        # Go up from core/utils/llm_config.py to project root
        project_root = current.parent.parent.parent
    
    config_dir = project_root / "config"
    env_file = config_dir / "env.backtest.yaml"
    
    if not env_file.exists():
        # Return default fallback configuration
        return _get_default_llm_config()
    
    try:
        # Load config with secrets merged in
        config = load_config_with_secrets(env_file)
        
        # Extract LLM section
        llm_config = config.get("llm", {})
        
        # If LLM section exists, return it with defaults merged
        if llm_config:
            default_config = _get_default_llm_config()
            # Merge defaults into provided config
            merged = default_config.copy()
            merged.update(llm_config)
            
            # Ensure nested dicts are merged properly
            if "model_capabilities" in llm_config:
                merged["model_capabilities"] = {
                    **default_config.get("model_capabilities", {}),
                    **llm_config["model_capabilities"],
                }
            
            return merged
        
        # No LLM section found, return defaults
        return _get_default_llm_config()
        
    except Exception as e:
        print(f"Warning: Could not load LLM config from {env_file}: {e}")
        return _get_default_llm_config()


def _get_default_llm_config() -> Dict[str, Any]:
    """Get default LLM configuration as fallback."""
    return {
        "primary_model": "gpt-5-mini",
        "fallback_models": [
            "gpt-5.1",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
        "max_completion_tokens": 2000,
        "timeout": 180.0,
        "model_capabilities": {},
    }


def get_llm_models(primary: Optional[str] = None, project_root: Optional[Path] = None) -> List[str]:
    """
    Get list of models to try (primary + fallbacks).
    
    Priority:
    1. OPENAI_MODEL environment variable
    2. Provided primary parameter
    3. primary_model from config
    4. Default "gpt-5-mini"
    
    Args:
        primary: Optional primary model name (overrides config)
        project_root: Optional project root path
    
    Returns:
        List of model names to try in order
    """
    # Check environment variable first (highest priority)
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        primary_model = env_model
    elif primary:
        primary_model = primary
    else:
        config = load_llm_config(project_root)
        primary_model = config.get("primary_model", "gpt-5-mini")
    
    # Get fallback models from config
    config = load_llm_config(project_root)
    fallbacks = config.get("fallback_models", [])
    
    # Remove primary from fallbacks if present
    fallbacks = [m for m in fallbacks if m != primary_model]
    
    return [primary_model] + fallbacks


def get_model_capabilities(model_name: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get capabilities for a specific model.
    
    Args:
        model_name: Name of the model (e.g., "gpt-5-mini")
        project_root: Optional project root path
    
    Returns:
        Dictionary of model capabilities, or empty dict if not found
    """
    config = load_llm_config(project_root)
    capabilities = config.get("model_capabilities", {})
    return capabilities.get(model_name, {})


def is_newer_model(model_name: str, project_root: Optional[Path] = None) -> bool:
    """
    Check if a model is a newer model (gpt-5, o1) that requires different parameters.
    
    Args:
        model_name: Name of the model
        project_root: Optional project root path
    
    Returns:
        True if model is newer (gpt-5 or o1), False otherwise
    """
    model_lower = model_name.lower()
    
    # Check config first for explicit capability
    capabilities = get_model_capabilities(model_name, project_root)
    if "supports_temperature" in capabilities:
        return not capabilities.get("supports_temperature", True)
    
    # Fallback to name-based detection
    return "gpt-5" in model_lower or "o1" in model_lower
