"""
Shared utilities for backtest scripts.

This module provides common functionality used across all backtest scripts to reduce
code duplication and ensure consistency.
"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from core.utils.config_loader import load_config_with_secrets
from core.backtest.scheduler import DecisionScheduler

logger = logging.getLogger(__name__)


def parse_date_string(date_str: str) -> date:
    """
    Parse a date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        date object
    
    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def load_backtest_config(
    project_root: Path,
    strategy_config_file: str,
    env_config_file: str = "env.backtest.yaml",
    strategy_name: Optional[str] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load and validate backtest configuration from env and strategy config files.
    
    Args:
        project_root: Path to project root directory
        strategy_config_file: Name of strategy config file (e.g., "strategy.llm_trend_detection.yaml")
        env_config_file: Name of environment config file (default: "env.backtest.yaml")
        strategy_name: Optional strategy name for config loading (some strategies need this)
    
    Returns:
        Tuple of (env_config, strategy_config, backtest_config)
    
    Raises:
        FileNotFoundError: If config files don't exist
        ValueError: If required config keys are missing
    """
    config_dir = project_root / "config"
    env_file = config_dir / env_config_file
    strategy_file = config_dir / strategy_config_file
    
    if not env_file.exists():
        raise FileNotFoundError(f"Config file not found: {env_file}")
    if not strategy_file.exists():
        raise FileNotFoundError(f"Strategy config file not found: {strategy_file}")
    
    # Load configs with secrets merged in
    env = load_config_with_secrets(env_file)
    if strategy_name:
        strat_cfg_raw = load_config_with_secrets(strategy_file, strategy_name=strategy_name)
    else:
        strat_cfg_raw = load_config_with_secrets(strategy_file)
    
    # Validate required config keys
    if "backtest" not in env:
        raise ValueError("Missing 'backtest' section in env.backtest.yaml")
    
    bt_cfg = env["backtest"]
    
    # Validate required backtest config keys
    required_bt_keys = ["start", "end", "initial_cash"]
    for key in required_bt_keys:
        if key not in bt_cfg:
            raise ValueError(f"Missing 'backtest.{key}' in env.backtest.yaml")
    
    return env, strat_cfg_raw, bt_cfg


def get_backtest_symbol(
    backtest_config: Dict,
    strategy_config: Dict,
    cli_symbol: Optional[str] = None,
) -> str:
    """
    Get symbol for backtest with priority: CLI arg > env.backtest.yaml > strategy config.
    
    Args:
        backtest_config: Backtest configuration dict from env.backtest.yaml
        strategy_config: Strategy configuration dict
        cli_symbol: Optional symbol from command-line argument (highest priority)
    
    Returns:
        Symbol string
    
    Raises:
        ValueError: If symbol is not found in any source
    """
    # Check for duplicate parameters and warn
    if "symbol" in backtest_config and "symbol" in strategy_config:
        bt_symbol = backtest_config["symbol"]
        strat_symbol = strategy_config["symbol"]
        if bt_symbol != strat_symbol:
            print(
                f"⚠️  WARNING: Duplicate 'symbol' parameter found:\n"
                f"   - env.backtest.yaml: '{bt_symbol}'\n"
                f"   - strategy config: '{strat_symbol}'\n"
                f"   Using '{bt_symbol}' from env.backtest.yaml (higher priority).\n"
                f"   Please remove 'symbol' from strategy config to avoid confusion."
            )
        else:
            print(
                f"⚠️  WARNING: Duplicate 'symbol' parameter found in both configs.\n"
                f"   Both have value '{bt_symbol}'. Please remove 'symbol' from strategy config.\n"
                f"   env.backtest.yaml takes priority."
            )
    
    # Priority: CLI arg > env.backtest.yaml > strategy config
    if cli_symbol:
        return cli_symbol
    elif "symbol" in backtest_config:
        return backtest_config["symbol"]
    elif "symbol" in strategy_config:
        return strategy_config["symbol"]
    else:
        raise ValueError(
            "Missing 'symbol'. Please provide --ticker, set 'backtest.symbol' in env.backtest.yaml, "
            "or 'symbol' in strategy config."
        )


def get_backtest_timeframe(
    backtest_config: Dict,
    strategy_config: Dict,
    default: str = "1D",
    cli_timeframe: Optional[str] = None,
) -> str:
    """
    Get timeframe for backtest with priority: CLI arg > env.backtest.yaml > strategy config > default.
    
    Args:
        backtest_config: Backtest configuration dict from env.backtest.yaml
        strategy_config: Strategy configuration dict
        default: Default timeframe if not found in configs
        cli_timeframe: Optional timeframe from command-line argument (highest priority)
    
    Returns:
        Timeframe string
    """
    # Check for duplicate parameters and warn
    if "timeframe" in backtest_config and "timeframe" in strategy_config:
        bt_timeframe = backtest_config["timeframe"]
        strat_timeframe = strategy_config["timeframe"]
        if bt_timeframe != strat_timeframe:
            print(
                f"⚠️  WARNING: Duplicate 'timeframe' parameter found:\n"
                f"   - env.backtest.yaml: '{bt_timeframe}'\n"
                f"   - strategy config: '{strat_timeframe}'\n"
                f"   Using '{bt_timeframe}' from env.backtest.yaml (higher priority).\n"
                f"   Please remove 'timeframe' from strategy config to avoid confusion."
            )
        else:
            print(
                f"⚠️  WARNING: Duplicate 'timeframe' parameter found in both configs.\n"
                f"   Both have value '{bt_timeframe}'. Please remove 'timeframe' from strategy config.\n"
                f"   env.backtest.yaml takes priority."
            )
    
    # Priority: CLI arg > env.backtest.yaml > strategy config > default
    if cli_timeframe:
        timeframe = cli_timeframe
        print(f"[Backtest] Using timeframe '{timeframe}' from command-line argument")
    else:
        timeframe = (
            backtest_config.get("timeframe")
            or strategy_config.get("timeframe")
            or default
        )
        
        # Log which config provided the timeframe
        if "timeframe" in backtest_config:
            print(f"[Backtest] Using timeframe '{timeframe}' from env.backtest.yaml")
        elif "timeframe" in strategy_config:
            print(
                f"[Backtest] Using timeframe '{timeframe}' from strategy config "
                "(env.backtest.yaml timeframe not set)"
            )
        else:
            print(f"[Backtest] Using default timeframe '{timeframe}'")
    
    return timeframe


def parse_backtest_dates(
    backtest_config: Dict,
    cli_start_date: Optional[str] = None,
    cli_end_date: Optional[str] = None,
    cli_days: Optional[int] = None,
) -> Tuple[datetime, datetime, float]:
    """
    Parse backtest start, end, and initial cash from config or CLI arguments.
    
    Priority: CLI explicit dates > --days parameter > env.backtest.yaml
    
    Args:
        backtest_config: Backtest configuration dict from env.backtest.yaml
        cli_start_date: Optional start date from CLI (YYYY-MM-DD format)
        cli_end_date: Optional end date from CLI (YYYY-MM-DD format)
        cli_days: Optional number of calendar days for backtest period
    
    Returns:
        Tuple of (start_datetime, end_datetime, initial_cash)
    
    Raises:
        ValueError: If date format is invalid, start > end, or days is invalid
    """
    # Priority: explicit dates > --days > config
    
    # Determine end date first
    if cli_end_date:
        end_date = parse_date_string(cli_end_date)
        end = datetime.combine(end_date, datetime.max.time())
    elif cli_days is not None:
        # If --days is specified, end date is today (or end of today)
        end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        end = datetime.fromisoformat(backtest_config["end"])
    
    # Determine start date
    if cli_start_date:
        # Explicit start date takes highest priority
        start_date = parse_date_string(cli_start_date)
        start = datetime.combine(start_date, datetime.min.time())
    elif cli_days is not None:
        # Calculate start date from end date minus days
        start = end - timedelta(days=cli_days)
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = datetime.fromisoformat(backtest_config["start"])
    
    # Validate date range
    if start > end:
        raise ValueError(f"Start date ({start.date()}) is after end date ({end.date()})")
    
    # Validate days parameter
    if cli_days is not None and cli_days <= 0:
        raise ValueError(f"Days parameter must be positive, got {cli_days}")
    
    initial_cash = backtest_config["initial_cash"]
    
    return start, end, initial_cash


def print_backtest_header(
    symbol: str,
    start: datetime,
    end: datetime,
    initial_cash: float,
    timeframe: str,
) -> None:
    """
    Print standardized backtest date range header.
    
    Args:
        symbol: Trading symbol
        start: Backtest start datetime
        end: Backtest end datetime
        initial_cash: Initial cash amount
        timeframe: Timeframe string
    """
    print("\n" + "=" * 80)
    print("BACKTEST DATE RANGE")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Start Date: {start.date()}")
    print(f"End Date:   {end.date()}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Timeframe: {timeframe}")
    print("=" * 80)
    print()


def create_scheduler_from_timeframe(timeframe: str) -> DecisionScheduler:
    """
    Create a DecisionScheduler based on timeframe string.
    
    Args:
        timeframe: Timeframe string (e.g., "1D", "1H", "15m", "30m")
    
    Returns:
        DecisionScheduler instance configured for the timeframe
    """
    t = timeframe.upper()
    if t in ("1D", "D"):
        # Daily: 24 hours = 1440 minutes
        return DecisionScheduler(interval_minutes=24 * 60)
    elif t.endswith("H"):
        # Hourly: parse number of hours (1H, 2H, 4H)
        try:
            hours = int(t[:-1]) if t != "H" else 1
        except ValueError:
            hours = 1
        return DecisionScheduler(interval_minutes=hours * 60)
    elif t.endswith("M") or t.endswith("m"):
        # Minutely: parse number of minutes (15m, 30m)
        try:
            minutes = int(t[:-1])
        except ValueError:
            minutes = 15
        return DecisionScheduler(interval_minutes=minutes)
    else:
        # Default to 15 minutes if parsing fails
        return DecisionScheduler(interval_minutes=15)
