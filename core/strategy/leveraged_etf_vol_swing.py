"""
Leveraged ETF Volatility Swing Trading Strategy

This strategy exploits daily and weekly price fluctuations in leveraged ETF pairs using:
- Trend filters (regime detection via underlying index)
- Volatility-based mean reversion
- Probability-driven targets
- Laddered execution

This strategy is NOT limited to SOXL/SOXS. It works with any bull/bear leveraged ETF pair
that tracks an underlying index. Examples:
- SOXL/SOXS (semiconductors, tracks SOXX)
- TQQQ/SQQQ (NASDAQ, tracks QQQ)
- UPRO/SPXU (S&P 500, tracks SPY)
- TNA/TZA (small caps, tracks IWM)
- LABU/LABD (biotech, tracks XBI)

Reference: Leveraged ETF Vol Swing Strategy document
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import math

from core.strategy.base import Strategy, Context
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.portfolio.portfolio import Position
from core.visualization.models import TradeSignal, IndicatorData, LeveragedETFIndicatorData


@dataclass
class LeveragedETFVolSwingConfig:
    """
    Configuration for Leveraged ETF Volatility Swing strategy.
    
    This strategy works with any bull/bear leveraged ETF pair. Examples:
    - SOXL/SOXS (semiconductors, regime: SOXX)
    - TQQQ/SQQQ (NASDAQ, regime: QQQ)
    - UPRO/SPXU (S&P 500, regime: SPY)
    - TNA/TZA (small caps, regime: IWM)
    - LABU/LABD (biotech, regime: XBI)
    """
    
    # Regime filter parameters
    # Underlying index used for regime detection (e.g., SOXX, QQQ, SPY, IWM, XBI)
    regime_symbol: str = "SOXX"  # Example: SOXX for semiconductors, QQQ for NASDAQ
    ema_fast: int = 20  # Fast EMA for regime detection
    ema_slow: int = 50  # Slow EMA for regime detection
    
    # Trading instruments
    # Bull leveraged ETF (3x long, e.g., SOXL, TQQQ, UPRO, TNA, LABU)
    bull_etf_symbol: str = "SOXL"  # Example: SOXL, TQQQ, UPRO, TNA, LABU
    # Bear leveraged ETF (3x short, e.g., SOXS, SQQQ, SPXU, TZA, LABD)
    bear_etf_symbol: str = "SOXS"  # Example: SOXS, SQQQ, SPXU, TZA, LABD
    
    # Indicator parameters
    bb_length: int = 20  # Bollinger Bands length
    bb_std: float = 2.0  # Bollinger Bands standard deviations
    atr_length: int = 14  # ATR period
    rsi_fast: int = 3  # Fast RSI period (2-3)
    rsi_slow: int = 14  # Slow RSI period
    volume_ma_length: int = 20  # Volume moving average length
    
    # Setup conditions
    rsi_fast_threshold: float = 10.0  # RSI fast must be < this
    rsi_slow_threshold: float = 45.0  # RSI slow must be < this
    volume_threshold: float = 0.7  # Volume must be >= 70% of 20-day average
    atr_support_range_min: float = 0.5  # Min ATR distance from support
    atr_support_range_max: float = 0.8  # Max ATR distance from support
    
    # Entry ladder parameters
    entry_ladder_1_pct: float = 0.6  # 60% of position at first entry
    entry_ladder_1_atr_offset: float = 0.3  # First entry: P_ref - 0.3×ATR
    entry_ladder_2_pct: float = 0.4  # 40% of position at second entry
    entry_ladder_2_atr_offset: float = 0.7  # Second entry: P_ref - 0.7×ATR
    
    # Exit parameters (ATR-based)
    # Note: stop_atr_multiple and max_holding_days are kept for config compatibility
    # but are no longer used in the strategy (stop loss and forced exit removed)
    stop_atr_multiple: float = 1.0  # Stop loss: -1 ATR (not used - removed from policy)
    target_1_atr_multiple: float = 1.0  # First target: +1 ATR (sell 60%)
    target_2_atr_multiple: float = 1.5  # Second target: +1.5 ATR (sell 40%)
    
    # Forced exit (not used - removed from policy)
    max_holding_days: int = 5  # Close position after N trading days (not used)
    
    # Position sizing
    risk_per_trade_pct: float = 0.75  # Risk 0.75% of equity per trade
    kelly_fraction: float = 0.25  # Use 0.25×Kelly for position sizing
    target_probability: float = 0.55  # Probability of hitting +1.5 ATR target (p2)
    
    # Weekly filter
    weekly_ma_length: int = 10  # 10-week MA for weekly filter
    
    # Timeframe
    timeframe: str = "1D"  # Daily bars
    
    # Capital deployment
    capital_deployment_pct: float = 1.0  # Percentage of available capital to deploy
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.capital_deployment_pct <= 1.0:
            raise ValueError(f"capital_deployment_pct must be between 0.0 and 1.0")
        if not 0.0 < self.risk_per_trade_pct <= 5.0:
            raise ValueError(f"risk_per_trade_pct must be between 0.0 and 5.0")
        if not 0.0 <= self.target_probability <= 1.0:
            raise ValueError(f"target_probability must be between 0.0 and 1.0")


class LeveragedETFVolSwingStrategy(Strategy):
    """
    Leveraged ETF Volatility Swing Trading Strategy.
    
    This strategy works with any bull/bear leveraged ETF pair. Examples:
    - SOXL/SOXS (semiconductors, regime: SOXX)
    - TQQQ/SQQQ (NASDAQ, regime: QQQ)
    - UPRO/SPXU (S&P 500, regime: SPY)
    - TNA/TZA (small caps, regime: IWM)
    - LABU/LABD (biotech, regime: XBI)
    
    Strategy Logic:
    1. Determine regime (Bull/Bear/Neutral) using underlying index EMA20/EMA50
    2. In Bull regime: Look for bull ETF long setups (mean reversion)
    3. In Bear regime: Look for bear ETF long setups (mean reversion)
    4. Use laddered entries and ATR-based exits
    5. Force exit after 5 trading days
    """
    
    def __init__(
        self,
        bull_etf_symbol: str,
        bear_etf_symbol: str,
        regime_symbol: str,
        config: LeveragedETFVolSwingConfig,
        data_engine: DataEngine,
    ):
        self._bull_etf_symbol = bull_etf_symbol  # e.g., SOXL, TQQQ, UPRO, TNA, LABU
        self._bear_etf_symbol = bear_etf_symbol  # e.g., SOXS, SQQQ, SPXU, TZA, LABD
        self._regime_symbol = regime_symbol  # e.g., SOXX, QQQ, SPY, IWM, XBI
        self._cfg = config
        self._data = data_engine
        
        # State tracking
        self._regime: Optional[str] = None  # "bull", "bear", "neutral"
        self._current_instrument: Optional[str] = None  # Current ETF being traded (bull_etf or bear_etf)
        self._entry_price: Optional[float] = None
        self._entry_date: Optional[datetime] = None
        self._atr_at_entry: Optional[float] = None
        self._position_size: Optional[int] = None
        self._ladder_1_filled: bool = False
        self._ladder_2_filled: bool = False
        self._target_1_hit: bool = False
        self._target_2_hit: bool = False
        
        # Indicator history for visualization
        self._signals: List[TradeSignal] = []
        self._indicator_history: List[LeveragedETFIndicatorData] = []
        self._bars_history: List[Bar] = []
        
        # Track bars for calculations
        self._regime_bars: List[Bar] = []  # Underlying index bars for regime detection
        self._bull_etf_bars: List[Bar] = []  # Bull ETF bars (e.g., SOXL, TQQQ, UPRO)
        self._bear_etf_bars: List[Bar] = []  # Bear ETF bars (e.g., SOXS, SQQQ, SPXU)
    
    async def on_start(self, ctx: Context) -> None:
        """Initialize strategy."""
        ctx.log(f"Leveraged ETF Volatility Swing Strategy started")
        ctx.log(f"  Regime symbol: {self._regime_symbol} (underlying index for regime detection)")
        ctx.log(f"  Bull ETF symbol: {self._bull_etf_symbol} (3x long leveraged ETF)")
        ctx.log(f"  Bear ETF symbol: {self._bear_etf_symbol} (3x short leveraged ETF)")
        ctx.log(f"  Timeframe: {self._cfg.timeframe}")
        ctx.log(f"  Risk per trade: {self._cfg.risk_per_trade_pct}%")
    
    async def on_decision(self, ctx: Context, now: datetime) -> None:
        """
        Main decision logic - called at each decision point.
        
        Strategy flow:
        1. Determine regime (Bull/Bear/Neutral) from underlying index
        2. Check for existing positions and manage exits
        3. Look for new entry setups if flat
        4. Execute laddered entries if setup triggers
        """
        # Get bars for calculations (need enough history for indicators)
        lookback_days = max(60, self._cfg.ema_slow + 20)  # Enough for EMA50 + indicators
        
        # Fetch regime bars (underlying index, e.g., SOXX, QQQ, SPY)
        # First try with requested lookback
        regime_bars = await self._data.get_bars(
            symbol=self._regime_symbol,
            start=now - timedelta(days=lookback_days),
            end=now,
            timeframe=self._cfg.timeframe,
        )
        
        # If insufficient data, try fetching from extended date to get all available data
        # This handles cases where backtest starts early in the available data
        # Use conservative 90-day lookback to avoid hitting API limits (e.g., MarketData.app 1-year limit)
        if len(regime_bars) < self._cfg.ema_slow:
            # Try fetching from 90 days ago (more conservative than 365 to avoid API limits)
            extended_start = now - timedelta(days=90)
            try:
                regime_bars = await self._data.get_bars(
                    symbol=self._regime_symbol,
                    start=extended_start,
                    end=now,
                    timeframe=self._cfg.timeframe,
                )
            except Exception as e:
                # If extended fetch fails (e.g., API limit), use what we have
                error_msg = str(e)
                if "402" in error_msg or "Payment Required" in error_msg or "1 year" in error_msg.lower():
                    # API limit hit - can't fetch more data
                    if not hasattr(self, '_insufficient_data_logged'):
                        self._insufficient_data_logged = set()
                    if self._regime_symbol not in self._insufficient_data_logged:
                        ctx.log(
                            f"Insufficient regime bars for {self._regime_symbol}: "
                            f"{len(regime_bars)} < {self._cfg.ema_slow} (minimum required). "
                            f"API data limit reached - cannot fetch more historical data."
                        )
                        self._insufficient_data_logged.add(self._regime_symbol)
                    return
                else:
                    # Other error - re-raise
                    raise
            
            # If still insufficient, log once and return
            if len(regime_bars) < self._cfg.ema_slow:
                # Only log once per symbol to avoid spam (use a simple flag)
                if not hasattr(self, '_insufficient_data_logged'):
                    self._insufficient_data_logged = set()
                
                if self._regime_symbol not in self._insufficient_data_logged:
                    ctx.log(
                        f"Insufficient regime bars for {self._regime_symbol}: "
                        f"{len(regime_bars)} < {self._cfg.ema_slow} (minimum required). "
                        f"Need more historical data to start strategy."
                    )
                    self._insufficient_data_logged.add(self._regime_symbol)
                return
        
        # Fetch bull and bear ETF bars (e.g., SOXL/SOXS, TQQQ/SQQQ, UPRO/SPXU)
        # Use the same start date as regime bars to ensure consistency
        # If regime bars were fetched from extended range, use that start date
        bars_start = regime_bars[0].timestamp if regime_bars else (now - timedelta(days=lookback_days))
        
        bull_etf_bars = await self._data.get_bars(
            symbol=self._bull_etf_symbol,
            start=bars_start,
            end=now,
            timeframe=self._cfg.timeframe,
        )
        bear_etf_bars = await self._data.get_bars(
            symbol=self._bear_etf_symbol,
            start=bars_start,
            end=now,
            timeframe=self._cfg.timeframe,
        )
        
        # If insufficient data, try extended range (conservative 90 days to avoid API limits)
        if len(bull_etf_bars) < self._cfg.atr_length or len(bear_etf_bars) < self._cfg.atr_length:
            extended_start = now - timedelta(days=90)
            try:
                bull_etf_bars = await self._data.get_bars(
                    symbol=self._bull_etf_symbol,
                    start=extended_start,
                    end=now,
                    timeframe=self._cfg.timeframe,
                )
                bear_etf_bars = await self._data.get_bars(
                    symbol=self._bear_etf_symbol,
                    start=extended_start,
                    end=now,
                    timeframe=self._cfg.timeframe,
                )
            except Exception as e:
                # If extended fetch fails (e.g., API limit), continue with what we have
                error_msg = str(e)
                if "402" in error_msg or "Payment Required" in error_msg or "1 year" in error_msg.lower():
                    # API limit hit - use what we have and check if it's sufficient
                    pass
                else:
                    # Other error - re-raise
                    raise
        
        if len(bull_etf_bars) < self._cfg.atr_length or len(bear_etf_bars) < self._cfg.atr_length:
            # Only log once per symbol to avoid spam
            if not hasattr(self, '_insufficient_data_logged'):
                self._insufficient_data_logged = set()
            
            missing_symbols = []
            if len(bull_etf_bars) < self._cfg.atr_length:
                missing_symbols.append(f"{self._bull_etf_symbol}={len(bull_etf_bars)}")
            if len(bear_etf_bars) < self._cfg.atr_length:
                missing_symbols.append(f"{self._bear_etf_symbol}={len(bear_etf_bars)}")
            
            log_key = f"{self._bull_etf_symbol}_{self._bear_etf_symbol}"
            if log_key not in self._insufficient_data_logged:
                ctx.log(
                    f"Insufficient bars: {', '.join(missing_symbols)} "
                    f"(minimum {self._cfg.atr_length} required). Need more historical data."
                )
                self._insufficient_data_logged.add(log_key)
            return
        
        # Update internal state
        self._regime_bars = regime_bars
        self._bull_etf_bars = bull_etf_bars
        self._bear_etf_bars = bear_etf_bars
        
        # 1. Determine regime
        regime = self._determine_regime(regime_bars)
        self._regime = regime
        
        # Store indicator data for visualization (regardless of position status)
        # This ensures BB values are available for all bars, including the most current day
        await self._store_indicator_data(ctx, now, bull_etf_bars, bear_etf_bars, regime)
        
        # Get current positions
        positions = ctx.portfolio.get_positions()
        current_position = None
        for pos in positions:
            if pos.symbol in [self._bull_etf_symbol, self._bear_etf_symbol]:
                current_position = pos
                break
        
        # 2. Manage existing positions
        if current_position:
            await self._manage_position(ctx, now, current_position, bull_etf_bars, bear_etf_bars)
        else:
            # 3. Look for new entry setups
            await self._check_entry_setups(ctx, now, bull_etf_bars, bear_etf_bars, regime)
    
    def _determine_regime(self, bars: List[Bar]) -> str:
        """
        Determine market regime using underlying index EMA20 and EMA50.
        
        Examples:
        - For SOXL/SOXS: uses SOXX bars
        - For TQQQ/SQQQ: uses QQQ bars
        - For UPRO/SPXU: uses SPY bars
        
        Returns:
            "bull": Price > EMA50 and EMA20 > EMA50
            "bear": Price < EMA50 and EMA20 < EMA50
            "neutral": Otherwise
        """
        if len(bars) < self._cfg.ema_slow:
            return "neutral"
        
        closes = [b.close for b in bars]
        
        # Calculate EMAs
        ema20 = self._calculate_ema(closes, self._cfg.ema_fast)
        ema50 = self._calculate_ema(closes, self._cfg.ema_slow)
        
        if len(ema20) == 0 or len(ema50) == 0:
            return "neutral"
        
        current_price = closes[-1]
        current_ema20 = ema20[-1]
        current_ema50 = ema50[-1]
        
        # Bull regime
        if current_price > current_ema50 and current_ema20 > current_ema50:
            return "bull"
        
        # Bear regime
        if current_price < current_ema50 and current_ema20 < current_ema50:
            return "bear"
        
        # Neutral
        return "neutral"
    
    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return []
        
        ema = []
        multiplier = 2.0 / (period + 1)
        
        # First EMA value is SMA
        sma = sum(values[:period]) / period
        ema.append(sma)
        
        # Subsequent values use EMA formula
        for i in range(period, len(values)):
            ema_value = (values[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_bollinger_bands(
        self, closes: List[float], length: int, std_dev: float
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands (upper, middle, lower).
        
        Formula:
        - Middle Band (BB Middle) = Simple Moving Average (SMA) of closing prices
        - Upper Band = SMA + (std_dev × standard deviation)
        - Lower Band = SMA - (std_dev × standard deviation)
        
        Args:
            closes: List of closing prices
            length: Period for SMA calculation (e.g., 20)
            std_dev: Number of standard deviations (e.g., 2.0)
        
        Returns:
            Tuple of (upper_bands, middle_bands, lower_bands) lists
            Note: Lists start at index (length - 1) since we need 'length' bars for first calculation
        """
        if len(closes) < length:
            return [], [], []
        
        middle = []  # SMA (Middle Band)
        upper = []
        lower = []
        
        for i in range(length - 1, len(closes)):
            window = closes[i - length + 1 : i + 1]
            sma = sum(window) / length
            
            # Calculate sample standard deviation (matches pandas rolling().std() which uses ddof=1)
            # This matches the FastAPI chart implementation
            if length > 1:
                variance = sum((x - sma) ** 2 for x in window) / (length - 1)  # Sample std: divide by n-1
            else:
                variance = 0.0
            std = math.sqrt(variance)
            
            middle.append(sma)
            upper.append(sma + std_dev * std)
            lower.append(sma - std_dev * std)
        
        return upper, middle, lower
    
    def _calculate_atr(self, bars: List[Bar], period: int) -> List[float]:
        """Calculate Average True Range."""
        if len(bars) < period + 1:
            return []
        
        true_ranges = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            true_ranges.append(tr)
        
        # Calculate ATR as SMA of True Range
        atr = []
        for i in range(period - 1, len(true_ranges)):
            window = true_ranges[i - period + 1 : i + 1]
            atr_value = sum(window) / period
            atr.append(atr_value)
        
        return atr
    
    def _calculate_rsi(self, closes: List[float], period: int) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return []
        
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        
        rsi = []
        for i in range(period - 1, len(gains)):
            avg_gain = sum(gains[i - period + 1 : i + 1]) / period
            avg_loss = sum(losses[i - period + 1 : i + 1]) / period
            
            if avg_loss == 0:
                rsi_value = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
            
            rsi.append(rsi_value)
        
        return rsi
    
    def _calculate_volume_ma(self, volumes: List[float], period: int) -> List[float]:
        """Calculate Volume Moving Average."""
        if len(volumes) < period:
            return []
        
        volume_ma = []
        for i in range(period - 1, len(volumes)):
            window = volumes[i - period + 1 : i + 1]
            ma_value = sum(window) / period
            volume_ma.append(ma_value)
        
        return volume_ma
    
    async def _store_indicator_data(
        self,
        ctx: Context,
        now: datetime,
        bull_etf_bars: List[Bar],
        bear_etf_bars: List[Bar],
        regime: str,
    ) -> None:
        """
        Store indicator data for visualization (called for every bar regardless of position status).
        This ensures BB values and other indicators are available for all bars, including the most current day.
        """
        # Determine which instrument to use for indicator data (use bull ETF for bull regime, bear ETF for bear regime)
        if regime == "bull":
            instrument = self._bull_etf_symbol
            bars = bull_etf_bars
        elif regime == "bear":
            instrument = self._bear_etf_symbol
            bars = bear_etf_bars
        else:
            # Neutral regime - use bull ETF as default
            instrument = self._bull_etf_symbol
            bars = bull_etf_bars
        
        if len(bars) < max(self._cfg.bb_length, self._cfg.atr_length, self._cfg.rsi_slow):
            return
        
        # Calculate indicators
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            closes, self._cfg.bb_length, self._cfg.bb_std
        )
        if not bb_lower:
            return
        
        # ATR
        atr = self._calculate_atr(bars, self._cfg.atr_length)
        if not atr:
            return
        
        # RSI
        rsi_fast = self._calculate_rsi(closes, self._cfg.rsi_fast)
        rsi_slow = self._calculate_rsi(closes, self._cfg.rsi_slow)
        if not rsi_fast or not rsi_slow:
            return
        
        # Volume MA
        volume_ma = self._calculate_volume_ma(volumes, self._cfg.volume_ma_length)
        if not volume_ma:
            return
        
        # Get current values (last in arrays)
        current_price = closes[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_middle = bb_middle[-1]
        current_bb_lower = bb_lower[-1]
        current_atr = atr[-1]
        current_rsi_fast = rsi_fast[-1]
        current_rsi_slow = rsi_slow[-1]
        current_volume = volumes[-1]
        current_volume_ma = volume_ma[-1]
        
        # Get current bar timestamp for indicator data
        current_bar_timestamp = bars[-1].timestamp if bars else now
        
        # Check if we already have indicator data for this timestamp (avoid duplicates)
        existing = any(
            ind.timestamp.date() == current_bar_timestamp.date() 
            for ind in self._indicator_history
        )
        
        if not existing:
            # Store indicator data for the current bar
            indicator_data = LeveragedETFIndicatorData(
                timestamp=current_bar_timestamp,
                price=current_price,
                bb_upper=current_bb_upper,
                bb_middle=current_bb_middle,
                bb_lower=current_bb_lower,
                atr=current_atr,
                rsi_fast=current_rsi_fast,
                rsi_slow=current_rsi_slow,
                volume=current_volume,
                volume_ma=current_volume_ma,
                regime=regime,
                entry_setup_detected=False,  # Will be set in _check_entry_setups if applicable
            )
            self._indicator_history.append(indicator_data)
    
    async def _check_entry_setups(
        self,
        ctx: Context,
        now: datetime,
        bull_etf_bars: List[Bar],
        bear_etf_bars: List[Bar],
        regime: str,
    ) -> None:
        """
        Check for entry setups based on regime.
        
        Examples:
        - Bull regime: Check SOXL, TQQQ, UPRO, TNA, LABU for long setups
        - Bear regime: Check SOXS, SQQQ, SPXU, TZA, LABD for long setups
        """
        
        # Determine which instrument to check
        if regime == "bull":
            instrument = self._bull_etf_symbol  # e.g., SOXL, TQQQ, UPRO
            bars = bull_etf_bars
        elif regime == "bear":
            instrument = self._bear_etf_symbol  # e.g., SOXS, SQQQ, SPXU
            bars = bear_etf_bars
        else:
            # Neutral regime - can trade both but at half size
            # For simplicity, skip neutral regime entries in initial implementation
            return
        
        if len(bars) < max(self._cfg.bb_length, self._cfg.atr_length, self._cfg.rsi_slow):
            return
        
        # Calculate indicators
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            closes, self._cfg.bb_length, self._cfg.bb_std
        )
        if not bb_lower:
            return
        
        # ATR
        atr = self._calculate_atr(bars, self._cfg.atr_length)
        if not atr:
            return
        
        # RSI
        rsi_fast = self._calculate_rsi(closes, self._cfg.rsi_fast)
        rsi_slow = self._calculate_rsi(closes, self._cfg.rsi_slow)
        if not rsi_fast or not rsi_slow:
            return
        
        # Volume MA
        volume_ma = self._calculate_volume_ma(volumes, self._cfg.volume_ma_length)
        if not volume_ma:
            return
        
        # Get current values (last in arrays)
        current_price = closes[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_middle = bb_middle[-1]
        current_bb_lower = bb_lower[-1]
        current_atr = atr[-1]
        current_rsi_fast = rsi_fast[-1]
        current_rsi_slow = rsi_slow[-1]
        current_volume = volumes[-1]
        current_volume_ma = volume_ma[-1]
        
        # Get current bar timestamp for indicator data
        current_bar_timestamp = bars[-1].timestamp if bars else now
        
        # Check setup conditions
        # 1. Location: Price touches lower BB or within ATR range of support
        # Relaxed: Allow price to be up to 1.2 ATR above BB lower (was 0.8)
        price_near_support = False
        if current_price <= current_bb_lower:
            price_near_support = True
        else:
            # Check if within ATR range of support (simplified - use BB lower as support)
            distance_from_support = current_price - current_bb_lower
            atr_distance_min = self._cfg.atr_support_range_min * current_atr
            # Extended max range to 1.2 ATR to be more lenient
            atr_distance_max = max(self._cfg.atr_support_range_max, 1.2) * current_atr
            if atr_distance_min <= distance_from_support <= atr_distance_max:
                price_near_support = True
            # Also allow if price is within 2% of BB lower (additional relaxed condition)
            elif distance_from_support <= current_bb_lower * 0.02:
                price_near_support = True
        
        # 2. Momentum exhaustion
        # Relaxed: Use OR logic - if either RSI is oversold, consider it OK
        # Also allow slightly higher RSI values
        rsi_fast_relaxed = self._cfg.rsi_fast_threshold * 1.5  # Allow up to 15 (was 10)
        rsi_slow_relaxed = self._cfg.rsi_slow_threshold * 1.2  # Allow up to 54 (was 45)
        rsi_ok = (
            (current_rsi_fast < rsi_fast_relaxed or current_rsi_slow < rsi_slow_relaxed)
            and current_rsi_slow < 60.0  # Still require slow RSI to be below 60
        )
        
        # 3. Volume
        # Relaxed: Lower threshold to 50% (was 70%)
        volume_threshold_relaxed = max(self._cfg.volume_threshold, 0.5)
        volume_ok = current_volume >= (volume_threshold_relaxed * current_volume_ma)
        
        # Log indicator values
        ctx.log(
            f"[{now.date()}] {instrument} | "
            f"Price={current_price:.2f} | "
            f"BB_Lower={current_bb_lower:.2f} | "
            f"ATR={current_atr:.2f} | "
            f"RSI_fast={current_rsi_fast:.1f} | "
            f"RSI_slow={current_rsi_slow:.1f} | "
            f"Vol={current_volume:,.0f} (MA={current_volume_ma:,.0f}) | "
            f"Regime={regime}"
        )
        
        # Log which conditions are met/failed for debugging
        # This helps identify why no signals are being generated
        conditions_status = []
        conditions_status.append(f"Price_near_support: {price_near_support} (Price={current_price:.2f}, BB_Lower={current_bb_lower:.2f}, diff={current_price - current_bb_lower:.2f})")
        conditions_status.append(f"RSI_ok: {rsi_ok} (RSI_fast={current_rsi_fast:.1f} < {self._cfg.rsi_fast_threshold}, RSI_slow={current_rsi_slow:.1f} < {self._cfg.rsi_slow_threshold})")
        conditions_status.append(f"Volume_ok: {volume_ok} (Vol={current_volume:,.0f} >= {self._cfg.volume_threshold * current_volume_ma:,.0f}, threshold={self._cfg.volume_threshold * current_volume_ma:,.0f})")
        
        # Check if conditions met
        # Use flexible logic: require at least 2 out of 3 conditions to be met
        # This allows signals when conditions are close but not perfect
        conditions_met = sum([price_near_support, rsi_ok, volume_ok])
        entry_setup_detected = conditions_met >= 2
        
        # Update existing indicator data with entry_setup_detected flag if it exists
        # (indicator data was already stored by _store_indicator_data)
        current_bar_date = current_bar_timestamp.date()
        for i, ind in enumerate(self._indicator_history):
            if ind.timestamp.date() == current_bar_date:
                # Replace with updated version that has entry_setup_detected flag
                updated_indicator_data = LeveragedETFIndicatorData(
                    timestamp=ind.timestamp,
                    price=ind.price,
                    bb_upper=ind.bb_upper,
                    bb_middle=ind.bb_middle,
                    bb_lower=ind.bb_lower,
                    atr=ind.atr,
                    rsi_fast=ind.rsi_fast,
                    rsi_slow=ind.rsi_slow,
                    volume=ind.volume,
                    volume_ma=ind.volume_ma,
                    regime=ind.regime,
                    entry_setup_detected=entry_setup_detected,
                )
                self._indicator_history[i] = updated_indicator_data
                break
        
        if entry_setup_detected:
            # Log which conditions were met
            met_conditions = []
            if price_near_support:
                met_conditions.append("Price near support")
            if rsi_ok:
                met_conditions.append("RSI oversold")
            if volume_ok:
                met_conditions.append("Volume sufficient")
            ctx.log(f"✅ ENTRY SETUP DETECTED for {instrument} in {regime} regime ({conditions_met}/3 conditions met: {', '.join(met_conditions)})")
            await self._execute_laddered_entry(ctx, now, instrument, current_price, current_atr, regime)
        else:
            # Log which conditions failed (only log occasionally to avoid spam)
            failed_conditions = []
            if not price_near_support:
                distance = current_price - current_bb_lower
                atr_min = self._cfg.atr_support_range_min * current_atr
                atr_max = max(self._cfg.atr_support_range_max, 1.2) * current_atr
                failed_conditions.append(f"Price not near support (Price-BB_Lower={distance:.2f}, need <=0 or {atr_min:.2f}-{atr_max:.2f})")
            if not rsi_ok:
                rsi_fast_relaxed = self._cfg.rsi_fast_threshold * 1.5
                rsi_slow_relaxed = self._cfg.rsi_slow_threshold * 1.2
                failed_conditions.append(f"RSI not oversold (fast={current_rsi_fast:.1f} need <{rsi_fast_relaxed:.1f}, slow={current_rsi_slow:.1f} need <{rsi_slow_relaxed:.1f})")
            if not volume_ok:
                volume_threshold_relaxed = max(self._cfg.volume_threshold, 0.5)
                failed_conditions.append(f"Volume too low ({current_volume:,.0f} < {volume_threshold_relaxed * current_volume_ma:,.0f})")
            
            # Only log every 10th bar to avoid spam, or if it's close to meeting conditions (1/3 or 2/3)
            should_log = (
                len(bars) % 10 == 0 or  # Every 10th bar
                conditions_met >= 1  # At least 1 condition met (getting close)
            )
            if should_log:
                ctx.log(f"  ⚠️  Entry conditions not met ({conditions_met}/3): {', '.join(failed_conditions)}")
    
    async def _execute_laddered_entry(
        self,
        ctx: Context,
        now: datetime,
        instrument: str,
        reference_price: float,
        atr: float,
        regime: str,
    ) -> None:
        """Execute laddered entry orders."""
        
        # Calculate position size
        # Get current equity by calculating portfolio value with current prices
        # Use current price as the market price for the instrument
        prices = {instrument: reference_price}
        # Also include prices for any existing positions
        positions = ctx.portfolio.get_positions()
        for pos in positions:
            if pos.symbol not in prices:
                # Use average price as fallback if no current price available
                prices[pos.symbol] = pos.avg_price
        
        equity = ctx.portfolio.equity(prices)
        risk_amount = equity * (self._cfg.risk_per_trade_pct / 100.0)
        
        # Position size based on risk percentage (stop loss removed from policy)
        # Use ATR as a volatility measure for position sizing, but don't enforce stop loss
        # Position size = risk_amount / (ATR as % of price) to normalize by volatility
        atr_pct = atr / reference_price
        position_value = risk_amount / atr_pct if atr_pct > 0 else risk_amount / 0.01  # Fallback to 1% if ATR is 0
        
        # Apply Kelly fraction if configured (updated to not use stop loss)
        if self._cfg.kelly_fraction > 0:
            p2 = self._cfg.target_probability
            # Kelly calculation without stop loss: simplified to target-based only
            kelly_f = (p2 * self._cfg.target_2_atr_multiple) / self._cfg.target_2_atr_multiple
            kelly_f = max(0, min(kelly_f, 1.0))  # Clamp between 0 and 1
            position_value *= (self._cfg.kelly_fraction * kelly_f)
        
        # Apply capital deployment percentage
        position_value *= self._cfg.capital_deployment_pct
        
        # Calculate share quantities
        total_shares = int(position_value / reference_price)
        if total_shares == 0:
            ctx.log(f"Position size too small: {position_value:.2f} / {reference_price:.2f} = {total_shares} shares")
            return
        
        ladder_1_shares = int(total_shares * self._cfg.entry_ladder_1_pct)
        ladder_2_shares = total_shares - ladder_1_shares  # Remaining shares
        
        # Calculate entry prices
        ladder_1_price = reference_price - (self._cfg.entry_ladder_1_atr_offset * atr)
        ladder_2_price = reference_price - (self._cfg.entry_ladder_2_atr_offset * atr)
        
        ctx.log(
            f"  Laddered Entry Plan:"
            f"  Total shares: {total_shares} | "
            f"  Ladder 1: {ladder_1_shares} shares @ ${ladder_1_price:.2f} (60%) | "
            f"  Ladder 2: {ladder_2_shares} shares @ ${ladder_2_price:.2f} (40%)"
        )
        
        # For backtesting, we'll use market orders at reference price
        # In live trading, these would be limit orders
        # Place first ladder entry (60%)
        order1 = Order(
            id=f"leveraged_etf_entry_1_{now.timestamp()}",
            symbol=instrument,
            side=Side.BUY,
            quantity=ladder_1_shares,
            order_type=OrderType.MARKET,  # In backtest, use market; live would be LIMIT
            limit_price=ladder_1_price if OrderType.LIMIT else None,
            instrument_type=InstrumentType.STOCK,
        )
        await ctx.execution.submit_order(order1)
        
        # Store entry state
        self._current_instrument = instrument
        self._entry_price = reference_price  # Use reference price for backtesting
        self._entry_date = now
        self._atr_at_entry = atr
        self._position_size = total_shares
        self._ladder_1_filled = True
        self._ladder_2_filled = False  # Will be filled on next bar if price drops
        
        # Create signal for visualization
        signal = TradeSignal(
            timestamp=now,
            price=reference_price,
            side="BUY",
            trend_score=1 if regime == "bull" else -1 if regime == "bear" else 0,
            di_plus=None,
            di_minus=None,
        )
        self._signals.append(signal)
        
        ctx.log(f"  ✅ Entry order placed: {ladder_1_shares} shares of {instrument} @ ${reference_price:.2f}")
    
    async def _manage_position(
        self,
        ctx: Context,
        now: datetime,
        position: Position,
        bull_etf_bars: List[Bar],
        bear_etf_bars: List[Bar],
    ) -> None:
        """Manage existing position: check targets (stop loss and forced exit removed)."""
        
        # Get current price
        if position.symbol == self._bull_etf_symbol:
            current_bars = bull_etf_bars
        else:
            current_bars = bear_etf_bars
        
        if not current_bars:
            return
        
        current_price = current_bars[-1].close
        
        # Check targets (ATR-based) - stop loss and forced exit removed
        if self._entry_price and self._atr_at_entry:
            entry = self._entry_price
            atr = self._atr_at_entry
            
            # Calculate targets (stop loss removed)
            target_1_price = entry + (self._cfg.target_1_atr_multiple * atr)
            target_2_price = entry + (self._cfg.target_2_atr_multiple * atr)
            
            # Check target 1 (sell 60% if not already hit)
            if not self._target_1_hit and current_price >= target_1_price:
                ctx.log(f"  🎯 Target 1 hit: {current_price:.2f} >= {target_1_price:.2f}")
                sell_quantity = int(position.quantity * self._cfg.entry_ladder_1_pct)
                await self._partial_close(ctx, position, sell_quantity, current_price, "target_1")
                self._target_1_hit = True
                return
            
            # Check target 2 (sell remaining 40% if target 1 already hit)
            if self._target_1_hit and not self._target_2_hit and current_price >= target_2_price:
                ctx.log(f"  🎯 Target 2 hit: {current_price:.2f} >= {target_2_price:.2f}")
                await self._close_position(ctx, position, current_price, "target_2")
                self._target_2_hit = True
                return
    
    async def _partial_close(
        self,
        ctx: Context,
        position: Position,
        quantity: int,
        price: float,
        reason: str,
    ) -> None:
        """Partially close a position."""
        if quantity <= 0 or quantity >= position.quantity:
            await self._close_position(ctx, position, price, reason)
            return
        
        order = Order(
            id=f"leveraged_etf_exit_{reason}_{position.symbol}_{datetime.now().timestamp()}",
            symbol=position.symbol,
            side=Side.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            limit_price=None,  # Market order doesn't need limit price
            instrument_type=InstrumentType.STOCK,
        )
        await ctx.execution.submit_order(order)
        
        # Create sell signal
        signal = TradeSignal(
            timestamp=datetime.now(),
            price=price,
            side="SELL",
            trend_score=0,  # Neutral for exits
            di_plus=None,
            di_minus=None,
        )
        self._signals.append(signal)
        
        ctx.log(f"  ✅ Partial exit: {quantity} shares @ ${price:.2f} ({reason})")
    
    async def _close_position(
        self,
        ctx: Context,
        position: Position,
        price: float,
        reason: str,
    ) -> None:
        """Close entire position."""
        order = Order(
            id=f"leveraged_etf_exit_{reason}_{position.symbol}_{datetime.now().timestamp()}",
            symbol=position.symbol,
            side=Side.SELL,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            limit_price=None,  # Market order doesn't need limit price
            instrument_type=InstrumentType.STOCK,
        )
        await ctx.execution.submit_order(order)
        
        # Create sell signal
        signal = TradeSignal(
            timestamp=datetime.now(),
            price=price,
            side="SELL",
            trend_score=0,  # Neutral for exits
            di_plus=None,
            di_minus=None,
        )
        self._signals.append(signal)
        
        # Reset position state
        self._current_instrument = None
        self._entry_price = None
        self._entry_date = None
        self._atr_at_entry = None
        self._position_size = None
        self._ladder_1_filled = False
        self._ladder_2_filled = False
        self._target_1_hit = False
        self._target_2_hit = False
        
        # Calculate P&L
        if position.avg_price:
            pnl = (price - position.avg_price) * position.quantity
            pnl_pct = ((price - position.avg_price) / position.avg_price) * 100
            ctx.log(
                f"  ✅ Position closed: {position.quantity} shares @ ${price:.2f} "
                f"(entry: ${position.avg_price:.2f}) | "
                f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
            )
    
    async def on_end(self, ctx: Context) -> None:
        """Cleanup when strategy ends."""
        ctx.log("Leveraged ETF Volatility Swing Strategy finished")
        
        # Close any remaining positions
        positions = ctx.portfolio.get_positions()
        for pos in positions:
            if pos.symbol in [self._bull_etf_symbol, self._bear_etf_symbol]:
                # Try to get the most recent price from cached bars
                # Use the last known bars from the strategy's internal state
                current_price = pos.avg_price  # Fallback to average entry price
                
                # Try to get price from recent bars (use last 5 days to find a trading day)
                try:
                    # Use the last known bars if available
                    if pos.symbol == self._bull_etf_symbol and hasattr(self, '_bull_etf_bars') and self._bull_etf_bars:
                        current_price = self._bull_etf_bars[-1].close
                    elif pos.symbol == self._bear_etf_symbol and hasattr(self, '_bear_etf_bars') and self._bear_etf_bars:
                        current_price = self._bear_etf_bars[-1].close
                    else:
                        # Try fetching recent data (but handle errors gracefully)
                        try:
                            bars = await self._data.get_bars(
                                symbol=pos.symbol,
                                start=self._entry_date if self._entry_date else (datetime.now() - timedelta(days=10)),
                                end=datetime.now(),
                                timeframe=self._cfg.timeframe,
                            )
                            if bars:
                                current_price = bars[-1].close
                        except Exception as e:
                            # If fetching fails (e.g., 404 for future dates), use average price
                            ctx.log(f"  ⚠️  Could not fetch current price for {pos.symbol}, using entry price: {e}")
                            current_price = pos.avg_price
                except Exception as e:
                    ctx.log(f"  ⚠️  Error getting price for {pos.symbol}, using entry price: {e}")
                    current_price = pos.avg_price
                
                await self._close_position(ctx, pos, current_price, "strategy_end")
    
    def get_signals(self) -> List[TradeSignal]:
        """Get all trade signals for visualization."""
        return self._signals
    
    def get_indicator_history(self) -> List[LeveragedETFIndicatorData]:
        """Get indicator history for visualization."""
        return self._indicator_history
    
    def get_bars(self) -> List[Bar]:
        """Get bars used by strategy."""
        return self._bars_history

