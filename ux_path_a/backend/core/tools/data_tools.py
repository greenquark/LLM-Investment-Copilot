"""
Data source tools for UX Path A.

These tools connect to the Trading Copilot Platform's data engine
to fetch market data. All data comes from the platform (INV-LLM-02).
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add project root to path BEFORE any imports to avoid conflicts
project_root = Path(__file__).parent.parent.parent.parent.parent
# Insert at the beginning to prioritize project root imports
# This ensures project root's 'core' package is found before backend's 'core' module
sys.path.insert(0, str(project_root))

# Try absolute import first (for local development), fallback to relative (for deployment)
try:
    from ux_path_a.backend.core.tools.registry import Tool
except ImportError:
    from core.tools.registry import Tool

# Import from project root's core package (not backend's core module)
# The project root is now first in sys.path, so 'core' should resolve to the project root's core package
try:
    from core.models.bar import Bar
except ImportError as e:
    # If that fails (e.g., backend's core.models.py shadows it), import directly from file
    import importlib.util
    bar_path = project_root / "core" / "models" / "bar.py"
    if bar_path.exists():
        # Import as a standalone module with a unique name to avoid conflicts
        spec = importlib.util.spec_from_file_location("_project_root_bar", str(bar_path))
        bar_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bar_module)
        Bar = bar_module.Bar
    else:
        raise ImportError(f"Could not find Bar model at {bar_path}. Original error: {e}")

# These should work since project root is in path
from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets

logger = logging.getLogger(__name__)


class GetSymbolDataTool(Tool):
    """
    Get symbol data including price, volume, and basic indicators.
    
    This tool fetches real market data from the platform (INV-LLM-02).
    """
    
    @property
    def name(self) -> str:
        return "get_symbol_data"
    
    @property
    def description(self) -> str:
        return (
            "Get current market data for a symbol including price, volume, "
            "and basic indicators. Returns real data from the platform. "
            "Use this to get up-to-date market information."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., 'AAPL', 'TSLA', 'SPY')",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for bars (e.g., '1D' for daily, '1h' for hourly)",
                    "default": "1D",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back",
                    "default": 30,
                },
            },
            "required": ["symbol"],
        }
    
    async def execute(self, symbol: str, timeframe: str = "1D", lookback_days: int = 30) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Load config and create data engine
            config_dir = project_root / "config"
            env_file = config_dir / "env.backtest.yaml"
            
            if not env_file.exists():
                return {
                    "error": "Configuration file not found",
                    "symbol": symbol,
                }
            
            env = load_config_with_secrets(env_file)
            data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
            
            # Calculate date range
            end = datetime.now()
            start = end - timedelta(days=lookback_days)
            
            # Fetch bars
            bars = await data_engine.get_bars(symbol, start, end, timeframe)
            
            if not bars:
                return {
                    "error": f"No data found for {symbol}",
                    "symbol": symbol,
                    "timeframe": timeframe,
                }
            
            # Get latest bar
            latest_bar = bars[-1]
            
            # Calculate basic statistics
            closes = [b.close for b in bars]
            volumes = [b.volume for b in bars]
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": latest_bar.close,
                "open": latest_bar.open,
                "high": latest_bar.high,
                "low": latest_bar.low,
                "volume": latest_bar.volume,
                "timestamp": latest_bar.timestamp.isoformat(),
                "price_change": closes[-1] - closes[0] if len(closes) > 1 else 0,
                "price_change_pct": ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 and closes[0] > 0 else 0,
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
                "bars_count": len(bars),
                "date_range": {
                    "start": bars[0].timestamp.isoformat(),
                    "end": bars[-1].timestamp.isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Error in get_symbol_data: {e}", exc_info=True)
            return {
                "error": str(e),
                "symbol": symbol,
            }


class GetBarsTool(Tool):
    """
    Get historical price bars for a symbol.
    
    Returns raw bar data for analysis.
    """
    
    @property
    def name(self) -> str:
        return "get_bars"
    
    @property
    def description(self) -> str:
        return (
            "Get historical price bars (OHLCV data) for a symbol. "
            "Returns raw bar data that can be used for technical analysis."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (defaults to today)",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe (e.g., '1D', '1h', '15m')",
                    "default": "1D",
                },
            },
            "required": ["symbol", "start_date"],
        }
    
    async def execute(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        timeframe: str = "1D",
    ) -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Load config
            config_dir = project_root / "config"
            env_file = config_dir / "env.backtest.yaml"
            
            if not env_file.exists():
                return {"error": "Configuration file not found"}
            
            env = load_config_with_secrets(env_file)
            data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
            
            # Parse dates
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date) if end_date else datetime.now()
            
            # Fetch bars
            bars = await data_engine.get_bars(symbol, start, end, timeframe)
            
            if not bars:
                return {
                    "error": f"No data found for {symbol}",
                    "symbol": symbol,
                }
            
            # Format bars for JSON
            bars_data = [
                {
                    "timestamp": b.timestamp.isoformat(),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars_data,
                "count": len(bars_data),
                "date_range": {
                    "start": bars[0].timestamp.isoformat(),
                    "end": bars[-1].timestamp.isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Error in get_bars: {e}", exc_info=True)
            return {"error": str(e)}


# Register tools
def register_data_tools(registry):
    """Register all data tools."""
    registry.register(GetSymbolDataTool())
    registry.register(GetBarsTool())
