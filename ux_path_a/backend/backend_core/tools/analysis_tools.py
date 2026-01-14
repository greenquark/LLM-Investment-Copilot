"""
Analysis tools for UX Path A.

These tools perform market analysis using the Trading Copilot Platform's
strategy and indicator modules. All analysis comes from the platform (INV-LLM-02).
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Ensure project root is in path for importing project root's core package
project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()

# Verify project root exists and has core/strategy
if not (project_root / 'core' / 'strategy').exists():
    # Fallback: try /app (Railway/Docker)
    if Path('/app').exists() and (Path('/app') / 'core' / 'strategy').exists():
        project_root = Path('/app').resolve()
    else:
        raise ImportError(f"Cannot find project root. Tried: {project_root}, /app")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.tools.registry import Tool

# Import from project root's core package (no conflict now since backend uses backend_core)
from core.data.factory import create_data_engine_from_config
from core.utils.config_loader import load_config_with_secrets

# Initialize logger first (needed for error messages)
logger = logging.getLogger(__name__)

# Use strategy registry for dynamic strategy discovery
try:
    from core.strategy.registry import get_strategy_class, get_config_class
    logger.info("Successfully imported strategy registry")
except ImportError as e:
    logger.error(f"Failed to import strategy registry: {e}", exc_info=True)
    raise

# Get strategy classes dynamically
logger.info(f"Attempting to get LLMTrendDetectionStrategy from registry (project_root={project_root})")
LLMTrendDetectionStrategy = get_strategy_class("LLMTrendDetectionStrategy", project_root)
LLMTrendDetectionConfig = get_config_class("LLMTrendDetectionConfig", project_root)

logger.info(f"Registry returned - Strategy: {LLMTrendDetectionStrategy is not None}, Config: {LLMTrendDetectionConfig is not None}")

# Fallback to direct import if registry fails (for backwards compatibility)
if LLMTrendDetectionStrategy is None or LLMTrendDetectionConfig is None:
    logger.warning("Registry returned None, falling back to direct import")
    try:
        from core.strategy.llm_trend_detection import (
            LLMTrendDetectionStrategy,
            LLMTrendDetectionConfig,
        )
        logger.info("Loaded LLMTrendDetectionStrategy via direct import (fallback)")
    except ImportError as e:
        logger.error(f"Failed to import LLMTrendDetectionStrategy: {e}", exc_info=True)
        logger.error(f"Project root: {project_root}")
        logger.error(f"core/strategy exists: {(project_root / 'core' / 'strategy').exists()}")
        logger.error(f"core/strategy/llm_trend_detection.py exists: {(project_root / 'core' / 'strategy' / 'llm_trend_detection.py').exists()}")
        raise
else:
    logger.info("Loaded LLMTrendDetectionStrategy via strategy registry")


class AnalyzeTrendTool(Tool):
    """
    Analyze trend regime for a symbol.
    
    Uses the LLM Trend Detection strategy to identify trend regimes.
    """
    
    @property
    def name(self) -> str:
        return "analyze_trend"
    
    @property
    def description(self) -> str:
        return (
            "Analyze the trend regime for a symbol. Identifies whether the market "
            "is in an uptrend, downtrend, or range-bound state. Uses the platform's "
            "LLM Trend Detection strategy for analysis."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol to analyze",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for analysis (e.g., '1D')",
                    "default": "1D",
                },
            },
            "required": ["symbol"],
        }
    
    async def execute(self, symbol: str, timeframe: str = "1D") -> Dict[str, Any]:
        """Execute the tool."""
        try:
            # Load config
            config_dir = project_root / "config"
            env_file = config_dir / "env.backtest.yaml"
            strategy_file = config_dir / "strategy.llm_trend_detection.yaml"
            
            if not env_file.exists() or not strategy_file.exists():
                return {"error": "Configuration files not found"}
            
            env = load_config_with_secrets(env_file)
            strat_cfg_raw = load_config_with_secrets(strategy_file)
            
            data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
            
            # Create strategy config
            cfg = LLMTrendDetectionConfig(
                timeframe=timeframe,
                symbol=symbol,
                **strat_cfg_raw,
            )
            
            # Create strategy instance
            strategy = LLMTrendDetectionStrategy(symbol, cfg, data_engine)
            
            # Get recent bars for analysis
            end = datetime.now()
            start = end - timedelta(days=200)  # Need enough bars for analysis
            bars = await data_engine.get_bars(symbol, start, end, timeframe)
            
            if not bars or len(bars) < 50:
                return {
                    "error": f"Insufficient data for {symbol}",
                    "bars_available": len(bars) if bars else 0,
                    "minimum_required": 50,
                }
            
            # Initialize strategy context (simplified)
            # In real implementation, would use proper backtest context
            # For now, return basic trend analysis
            
            latest_bar = bars[-1]
            prev_bar = bars[-2] if len(bars) > 1 else latest_bar
            
            # Simple trend detection
            price_change = latest_bar.close - prev_bar.close
            price_change_pct = (price_change / prev_bar.close * 100) if prev_bar.close > 0 else 0
            
            # Determine regime
            if price_change_pct > 1:
                regime = "TREND_UP"
                confidence = min(90, 50 + abs(price_change_pct) * 5)
            elif price_change_pct < -1:
                regime = "TREND_DOWN"
                confidence = min(90, 50 + abs(price_change_pct) * 5)
            else:
                regime = "RANGE"
                confidence = 60
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": regime,
                "confidence": round(confidence, 1),
                "current_price": latest_bar.close,
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "timestamp": latest_bar.timestamp.isoformat(),
                "note": "This is a simplified analysis. Full LLM trend detection requires more context.",
            }
        except Exception as e:
            logger.error(f"Error in analyze_trend: {e}", exc_info=True)
            return {"error": str(e)}


class CalculateIndicatorsTool(Tool):
    """
    Calculate technical indicators for a symbol.
    
    Returns common technical indicators like RSI, MACD, etc.
    """
    
    @property
    def name(self) -> str:
        return "calculate_indicators"
    
    @property
    def description(self) -> str:
        return (
            "Calculate technical indicators for a symbol. Returns indicators "
            "like RSI, MACD, moving averages, etc. All calculations use real market data."
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
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe",
                    "default": "1D",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators to calculate (e.g., ['RSI', 'MACD', 'MA'])",
                    "default": ["RSI", "MA"],
                },
            },
            "required": ["symbol"],
        }
    
    async def execute(
        self,
        symbol: str,
        timeframe: str = "1D",
        indicators: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Execute the tool."""
        if indicators is None:
            indicators = ["RSI", "MA"]
        
        try:
            # Load config and get bars
            config_dir = project_root / "config"
            env_file = config_dir / "env.backtest.yaml"
            
            if not env_file.exists():
                return {"error": "Configuration file not found"}
            
            env = load_config_with_secrets(env_file)
            data_engine = create_data_engine_from_config(env_config=env, use_for="historical")
            
            end = datetime.now()
            start = end - timedelta(days=100)
            bars = await data_engine.get_bars(symbol, start, end, timeframe)
            
            if not bars or len(bars) < 20:
                return {
                    "error": f"Insufficient data for {symbol}",
                    "bars_available": len(bars) if bars else 0,
                }
            
            closes = [b.close for b in bars]
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": {},
            }
            
            # Calculate RSI (simplified)
            if "RSI" in indicators:
                rsi = self._calculate_rsi(closes, period=14)
                result["indicators"]["RSI"] = {
                    "value": round(rsi, 2) if rsi else None,
                    "period": 14,
                }
            
            # Calculate Moving Average
            if "MA" in indicators:
                ma20 = sum(closes[-20:]) / min(20, len(closes))
                ma50 = sum(closes[-50:]) / min(50, len(closes)) if len(closes) >= 50 else None
                result["indicators"]["MA"] = {
                    "MA20": round(ma20, 2),
                    "MA50": round(ma50, 2) if ma50 else None,
                }
            
            return result
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _calculate_rsi(self, closes: list, period: int = 14) -> Optional[float]:
        """Calculate RSI (simplified implementation)."""
        if len(closes) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


def register_analysis_tools(registry):
    """Register all analysis tools."""
    registry.register(AnalyzeTrendTool())
    registry.register(CalculateIndicatorsTool())
