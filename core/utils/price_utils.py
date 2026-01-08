"""
Utility functions for fetching and calculating prices.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Dict, Optional

from core.data.base import DataEngine


async def get_final_price(
    data_engine: DataEngine,
    symbol: str,
    target_date: date,
    timeframe: str,
    equity_curve: Optional[Dict[datetime, float]] = None,
    final_shares: float = 0.0,
    fallback_to_equity_curve: bool = True,
) -> float:
    """
    Get the final price for a symbol on a given date.
    
    Args:
        data_engine: Data engine to fetch bars from
        symbol: Stock symbol
        target_date: Date to fetch price for
        timeframe: Timeframe for bars (e.g., "1D")
        equity_curve: Optional equity curve for fallback calculation
        final_shares: Number of shares held (for equity curve fallback)
        fallback_to_equity_curve: Whether to use equity curve as fallback
    
    Returns:
        Final price (close price of the bar), or 0.0 if not found
    """
    try:
        day_start = datetime.combine(target_date, datetime.min.time())
        day_end = datetime.combine(target_date, datetime.max.time())
        final_bars = await data_engine.get_bars(
            symbol,
            day_start,
            day_end,
            timeframe=timeframe
        )
        if final_bars:
            return final_bars[-1].close
        elif fallback_to_equity_curve and equity_curve:
            # Fallback: use last equity curve value to estimate price
            equity_items = sorted(equity_curve.items(), key=lambda x: x[0])
            if equity_items and final_shares > 0:
                return equity_items[-1][1] / final_shares
        return 0.0
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not fetch final price for {symbol} on {target_date}: {e}")
        return 0.0

