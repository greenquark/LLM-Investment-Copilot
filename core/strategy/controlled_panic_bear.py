"""
Controlled Panic Bear Strategy

A bear market strategy that enters positions during panic conditions:
- High VIX (volatility spike)
- CNN Fear & Greed Index in panic range
- Price below SMA20 with negative 3-day return
- Uses options strategies (put spreads, diagonals, lotto puts)


It is a regime-based volatility + downside capture strategy:

Market state	                    Action
Normal / greedy / low vol	        Do nothing
Fear rising, volatility expanding	Enter bearish exposure
Panic peak or stabilization	        Exit / reduce

The strategy tries to:

Avoid bleed during calm markets
Exploit convexity when fear spikes
Limit tail risk via spreads / defined-risk structures
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.data.fear_greed_index import get_fgi_value
from core.models.bar import Bar


@dataclass
class ControlledPanicBearConfig:
    """Configuration for Controlled Panic Bear strategy."""
    
    # VIX thresholds
    vix_min: float = 20.0
    vix_mult: float = 1.2  # VIX must be >= vix_ma10 * vix_mult
    
    # CNN Fear & Greed thresholds
    cnn_min: float = 5.0
    cnn_max: float = 25.0
    
    # Price action thresholds
    ret3d_min: float = -0.03  # Minimum 3-day return (negative)
    
    # Position sizing (as fraction of portfolio)
    core_size: float = 0.015  # Put spread size
    diag_size: float = 0.01   # Diagonal size
    lotto_size: float = 0.01   # Lotto put size
    
    # Maximum exposure limits
    max_core_exposure: float = 0.06
    max_diag_exposure: float = 0.04
    max_lotto_exposure: float = 0.02
    max_total_exposure: float = 0.08
    
    @classmethod
    def from_dict(cls, config: dict) -> "ControlledPanicBearConfig":
        """Create config from dictionary."""
        return cls(
            vix_min=config.get("vix_min", 20.0),
            vix_mult=config.get("vix_mult", 1.2),
            cnn_min=config.get("cnn_min", 5.0),
            cnn_max=config.get("cnn_max", 25.0),
            ret3d_min=config.get("ret3d_min", -0.03),
            core_size=config.get("core_size", 0.015),
            diag_size=config.get("diag_size", 0.01),
            lotto_size=config.get("lotto_size", 0.01),
            max_core_exposure=config.get("max_core_exposure", 0.06),
            max_diag_exposure=config.get("max_diag_exposure", 0.04),
            max_lotto_exposure=config.get("max_lotto_exposure", 0.02),
            max_total_exposure=config.get("max_total_exposure", 0.08),
        )


class ControlledPanicBearStrategy(Strategy):
    """
    Controlled Panic Bear Strategy.
    
    Enters bearish positions during panic conditions:
    - High VIX (volatility spike)
    - CNN Fear & Greed in panic range
    - Price below SMA20 with negative returns
    
    Exits when:
    - Price above SMA20
    - CNN Fear & Greed >= 35 (greed)
    - VIX < 18 (low volatility)
    """
    
    def __init__(
        self,
        symbol: str,
        config: ControlledPanicBearConfig,
        data_engine: DataEngine,
        vix_symbol: str = "^VIX",  # VIX symbol
    ):
        """
        Initialize the strategy.
        
        Args:
            symbol: Underlying symbol (e.g., "SPY", "SPX")
            config: Strategy configuration
            data_engine: Data engine for fetching market data
            vix_symbol: VIX symbol (default "^VIX")
        """
        self.symbol = symbol
        self.config = config
        self.data_engine = data_engine
        self.vix_symbol = vix_symbol
        
        # Strategy state
        self._initialized = False
        self._bars: List[Bar] = []
        self._vix_bars: List[Bar] = []
        self._exposure: Dict[str, float] = {
            "core": 0.0,
            "diag": 0.0,
            "lotto": 0.0,
            "total": 0.0,
        }
        self._intents: List[Dict] = []
    
    async def on_start(self, ctx: Context) -> None:
        """Initialize the strategy."""
        ctx.log(f"Starting Controlled Panic Bear strategy for {self.symbol}")
        ctx.log(f"Config: vix_min={self.config.vix_min}, cnn_range=[{self.config.cnn_min}, {self.config.cnn_max}]")
        
        # Note about FGI data availability
        ctx.log("FGI data: Using core.data.fear_greed_index module")
        ctx.log("  - Current values: fear-and-greed package")
        ctx.log("  - Historical values: CNN API (https://production.dataviz.cnn.io)")
        
        self._initialized = True
        ctx.log("Strategy initialized")
    
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """Make trading decisions based on current market conditions."""
        if not self._initialized:
            ctx.log("Warning: Strategy not initialized, skipping decision")
            return
        
        try:
            # Fetch recent bars for underlying
            lookback_start = now - timedelta(days=30)
            bars = await self.data_engine.get_bars(
                self.symbol, lookback_start, now, timeframe="1D"
            )
            if not bars:
                ctx.log(f"No bars available for {self.symbol}")
                return
            
            self._bars = bars
            latest_bar = bars[-1]
            
            # Fetch VIX data
            vix_bars = await self.data_engine.get_bars(
                self.vix_symbol, lookback_start, now, timeframe="1D"
            )
            if not vix_bars:
                ctx.log(f"No VIX bars available for {self.vix_symbol}")
                return
            
            self._vix_bars = vix_bars
            
            # Calculate indicators
            snapshot = self._create_snapshot(latest_bar, vix_bars, now)
            if snapshot is None:
                ctx.log("Could not create snapshot (insufficient data)")
                return
            
            # Check exit conditions first
            if self._should_close(snapshot):
                ctx.log(f"Exit signal: close={snapshot['close']:.2f}, SMA20={snapshot['sma20']:.2f}, "
                       f"CNN={snapshot['cnn']}, VIX={snapshot['vix']:.2f}")
                # Close all positions
                # Note: Actual options position closing would be handled by execution engine
                # This is a placeholder for the strategy logic
                total_exposure = self._exposure.get("total", 0.0)
                if total_exposure > 0:
                    # In a real implementation, this would close options positions
                    # For now, we just reset exposure tracking
                    self._exposure = {"core": 0.0, "diag": 0.0, "lotto": 0.0, "total": 0.0}
                    ctx.log(f"Exit signal: Closing all positions (exposure was {total_exposure:.2%})")
                    # TODO: Implement actual options position closing via execution engine
                return
            
            # Check entry conditions
            if self._can_enter(snapshot):
                ctx.log(f"Entry signal detected: VIX={snapshot['vix']:.2f}, CNN={snapshot['cnn']}, "
                       f"ret3d={snapshot['ret3d']:.2%}")
                
                # Generate trading intents
                intents = self._generate_intents(snapshot, now)
                for intent in intents:
                    ctx.log(f"Intent: {intent['type']} size={intent['size_pct']:.2%} params={intent['params']}")
                    self._intents.append(intent)
                    
                    # Update exposure tracking (simplified - actual execution would be handled by execution engine)
                    if intent['type'] == 'PUT_SPREAD':
                        self._exposure['core'] += intent['size_pct']
                    elif intent['type'] == 'DIAGONAL':
                        self._exposure['diag'] += intent['size_pct']
                    elif intent['type'] == 'LOTTO_PUT':
                        self._exposure['lotto'] += intent['size_pct']
                    
                    self._exposure['total'] = (
                        self._exposure['core'] + 
                        self._exposure['diag'] + 
                        self._exposure['lotto']
                    )
            else:
                ctx.log(f"No entry signal: VIX={snapshot['vix']:.2f}, CNN={snapshot['cnn']}, "
                       f"price_ok={snapshot['close'] < snapshot['sma20']}, ret3d={snapshot['ret3d']:.2%}")
        
        except Exception as e:
            ctx.log(f"Error in on_decision: {e}")
            import traceback
            ctx.log(traceback.format_exc())
    
    async def on_end(self, ctx: Context) -> None:
        """Clean up when strategy ends."""
        ctx.log(f"Ending Controlled Panic Bear strategy for {self.symbol}")
        
        # Log final exposure
        ctx.log(f"Final exposure: {self._exposure}")
        ctx.log(f"Total intents generated: {len(self._intents)}")
        
        # Log portfolio state
        # Get current price from last bar if available, otherwise use cash only
        if self._bars:
            current_price = self._bars[-1].close
            prices = {self.symbol: current_price}
            portfolio_value = ctx.portfolio.equity(prices)
        else:
            # No bars available, just use cash
            portfolio_value = ctx.portfolio.state.cash
        ctx.log(f"Final portfolio value: ${portfolio_value:,.2f}")
    
    def _create_snapshot(self, bar: Bar, vix_bars: List[Bar], now: datetime) -> Optional[Dict]:
        """
        Create a snapshot of current market conditions.
        
        Returns:
            Dictionary with snapshot data or None if insufficient data
        """
        if len(self._bars) < 20:
            return None
        
        # Calculate SMA20
        recent_closes = [b.close for b in self._bars[-20:]]
        sma20 = sum(recent_closes) / len(recent_closes)
        
        # Calculate 3-day return
        if len(self._bars) >= 4:
            close_3d_ago = self._bars[-4].close
            ret3d = (bar.close - close_3d_ago) / close_3d_ago
        else:
            ret3d = 0.0
        
        # Get VIX and VIX MA10
        if len(vix_bars) < 10:
            return None
        
        vix = vix_bars[-1].close
        vix_closes = [b.close for b in vix_bars[-10:]]
        vix_ma10 = sum(vix_closes) / len(vix_closes)
        
        # Get CNN Fear & Greed Index using shared module
        # This will automatically use CNN API for historical dates and fear-and-greed package for current dates
        cnn_value, _ = get_fgi_value(target_date=bar.timestamp.date())
        cnn = cnn_value  # Use the value (or None if unavailable)
        
        return {
            "as_of": bar.timestamp.date(),
            "underlying": self.symbol,
            "close": bar.close,
            "sma20": sma20,
            "ret3d": ret3d,
            "vix": vix,
            "vix_ma10": vix_ma10,
            "cnn": cnn,
        }
    
    def _should_close(self, snapshot: Dict) -> bool:
        """Check if positions should be closed."""
        close_above_sma = snapshot["close"] > snapshot["sma20"]
        cnn_greed = (snapshot["cnn"] or 0) >= 35
        vix_low = snapshot["vix"] < 18
        
        return close_above_sma and cnn_greed and vix_low
    
    def _can_enter(self, snapshot: Dict) -> bool:
        """Check if entry conditions are met."""
        # VIX conditions
        vix_ok = (
            snapshot["vix"] >= self.config.vix_min and
            snapshot["vix"] >= snapshot["vix_ma10"] * self.config.vix_mult
        )
        
        # CNN Fear & Greed conditions
        cnn_ok = (
            snapshot["cnn"] is not None and
            self.config.cnn_min <= snapshot["cnn"] <= self.config.cnn_max
        )
        
        # Price action conditions
        price_ok = (
            snapshot["close"] < snapshot["sma20"] and
            snapshot["ret3d"] <= self.config.ret3d_min
        )
        
        return vix_ok and cnn_ok and price_ok

    def _generate_intents(self, snapshot: Dict, now: datetime) -> List[Dict]:
        """Generate trading intents based on current exposure and conditions."""
        intents = []
        
        # Core put spread
        if (self._exposure.get("core", 0) < self.config.max_core_exposure and
            self._exposure.get("total", 0) < self.config.max_total_exposure):
            intents.append({
                "as_of": snapshot["as_of"],
                "type": "PUT_SPREAD",
                "size_pct": self.config.core_size,
                "params": {
                    "dte": 60,
                    "long_pct_otm": 0.03,
                    "short_pct_otm": 0.08,
                }
            })
        
        # Diagonal spread
        if (self._exposure.get("diag", 0) < self.config.max_diag_exposure and
            self._exposure.get("total", 0) < self.config.max_total_exposure):
            intents.append({
                "as_of": snapshot["as_of"],
                "type": "DIAGONAL",
                "size_pct": self.config.diag_size,
                "params": {
                    "long_dte": 70,
                    "short_dte": 20,
                    "long_otm": 0.04,
                    "short_otm": 0.07,
                }
            })
        
        # Lotto puts (only on events - simplified to check if CNN is very low)
        is_event = snapshot["cnn"] is not None and snapshot["cnn"] <= 10
        if (is_event and
            self._exposure.get("lotto", 0) < self.config.max_lotto_exposure):
            intents.append({
                "as_of": snapshot["as_of"],
                "type": "LOTTO_PUT",
                "size_pct": self.config.lotto_size,
                "params": {
                    "dte": 25,
                    "otm": 0.04,
                }
            })
        
        return intents
    
    def get_intents(self) -> List[Dict]:
        """Get all generated intents."""
        return self._intents
    
    def get_exposure(self) -> Dict[str, float]:
        """Get current exposure levels."""
        return self._exposure.copy()
