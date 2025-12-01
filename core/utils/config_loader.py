"""
Configuration loader with secrets support.

This module provides utilities to load configuration files and merge
secrets from a separate secrets.yaml file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_secrets(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load secrets from config/secrets.yaml if it exists.
    
    Args:
        config_dir: Directory containing config files. If None, assumes
                   config/ directory relative to project root.
    
    Returns:
        Dictionary of secrets, or empty dict if secrets file doesn't exist.
    """
    if config_dir is None:
        # Try to find project root by looking for common markers
        current = Path(__file__).resolve()
        # Go up from core/utils/config_loader.py to project root
        project_root = current.parent.parent.parent
        config_dir = project_root / "config"
    else:
        config_dir = Path(config_dir)
    
    secrets_file = config_dir / "secrets.yaml"
    
    if not secrets_file.exists():
        return {}
    
    try:
        with open(secrets_file) as f:
            secrets = yaml.safe_load(f) or {}
        return secrets
    except Exception as e:
        print(f"Warning: Could not load secrets from {secrets_file}: {e}")
        return {}


def merge_secrets_into_config(
    config: Dict[str, Any],
    secrets: Dict[str, Any],
    strategy_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge secrets into configuration dictionary.
    
    This function:
    1. Merges data_sources secrets (e.g., api_token) into config["data_sources"]
    2. Merges strategy-specific secrets (e.g., openai_api_key) into strategy config
    3. Merges moomoo secrets into config["moomoo"] if present
    
    Args:
        config: Main configuration dictionary
        secrets: Secrets dictionary from load_secrets()
        strategy_name: Name of strategy (e.g., "llm_trend_detection") for strategy-specific secrets
    
    Returns:
        Updated configuration dictionary with secrets merged in
    """
    if not secrets:
        return config
    
    # Make a copy to avoid modifying the original
    merged = config.copy()
    
    # Merge data_sources secrets
    if "data_sources" in secrets and "data_sources" in merged:
        for source_name, source_secrets in secrets["data_sources"].items():
            if source_name in merged["data_sources"]:
                if isinstance(source_secrets, dict) and isinstance(merged["data_sources"][source_name], dict):
                    # Merge secrets into existing data source config
                    merged["data_sources"][source_name].update(source_secrets)
                elif "api_token" in source_secrets:
                    # If source_secrets is just a dict with api_token, merge it
                    if isinstance(merged["data_sources"][source_name], dict):
                        merged["data_sources"][source_name]["api_token"] = source_secrets["api_token"]
    
    # Merge strategy-specific secrets
    if strategy_name and "strategies" in secrets:
        strategy_secrets = secrets["strategies"].get(strategy_name, {})
        if strategy_secrets:
            # Merge strategy secrets into the config dict
            merged.update(strategy_secrets)
    
    # Merge moomoo secrets
    if "moomoo" in secrets and "moomoo" in merged:
        if isinstance(secrets["moomoo"], dict) and isinstance(merged["moomoo"], dict):
            merged["moomoo"].update(secrets["moomoo"])
    
    return merged


def load_config_with_secrets(
    config_file: Path,
    secrets_file: Optional[Path] = None,
    strategy_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a configuration file and merge secrets into it.
    
    This is a convenience function that:
    1. Loads the main config file
    2. Loads secrets (from secrets_file or default location)
    3. Merges secrets into the config
    4. Returns the merged config
    
    Args:
        config_file: Path to the main configuration YAML file
        secrets_file: Optional path to secrets file. If None, uses config/secrets.yaml
        strategy_name: Optional strategy name for strategy-specific secrets
    
    Returns:
        Configuration dictionary with secrets merged in
    """
    # Load main config
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
    
    # Load secrets
    if secrets_file:
        with open(secrets_file) as f:
            secrets = yaml.safe_load(f) or {}
    else:
        secrets = load_secrets(config_file.parent)
    
    # Merge secrets into config
    return merge_secrets_into_config(config, secrets, strategy_name)

