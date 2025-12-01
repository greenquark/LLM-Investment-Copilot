from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from core.models.fill import Fill
from core.models.position import Position
from core.models.portfolio_state import PortfolioState

@dataclass
class Portfolio:
    state: PortfolioState

    def apply_fill(self, fill: Fill):
        """
        Apply a fill to the portfolio, updating positions and cash accurately.
        
        For BUY: quantity is positive, we add to position and reduce cash
        For SELL: quantity is negative, we reduce position and increase cash
        """
        pos = self.state.positions.get(fill.symbol)
        qty = fill.quantity  # Signed quantity: positive for BUY, negative for SELL
        
        if pos is None:
            # Opening a new position (must be a BUY, so qty > 0)
            if qty <= 0:
                raise ValueError(f"Cannot open position with non-positive quantity: {qty}")
            self.state.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=qty,
                avg_price=fill.price,
                instrument_type="STOCK",
            )
        else:
            new_qty = pos.quantity + qty
            if new_qty == 0:
                # Closing the entire position
                # Realized P&L is automatically reflected in cash update below:
                # - We originally paid: avg_price * quantity (reduced cash)
                # - We now receive: fill.price * quantity (increases cash)
                # - Net P&L = (fill.price - avg_price) * quantity
                del self.state.positions[fill.symbol]
            elif new_qty < 0:
                raise ValueError(f"Position quantity cannot be negative: {new_qty}")
            elif qty > 0:
                # Adding to position (BUY more shares)
                # Calculate weighted average cost basis
                cost_before = pos.avg_price * pos.quantity
                cost_added = fill.price * qty
                pos.quantity = new_qty
                pos.avg_price = (cost_before + cost_added) / new_qty
            else:
                # Reducing position (SELL some shares)
                # Average price of remaining shares stays the same (FIFO/average cost basis)
                # We don't recalculate avg_price when selling - it remains the original cost basis
                pos.quantity = new_qty
                # avg_price remains unchanged for remaining shares

        # Update cash: BUY reduces cash, SELL increases cash
        # fill.quantity is positive for BUY, negative for SELL
        # So: cash -= (price * quantity + commission)
        # For BUY: cash -= (price * positive + commission) = cash decreases ✓
        # For SELL: cash -= (price * negative + commission) = cash += (price * positive) - commission = cash increases ✓
        # When closing a position, the realized P&L is: (sell_price - buy_avg_price) * quantity
        # This P&L is automatically reflected in the cash increase
        self.state.cash -= fill.price * fill.quantity + fill.commission

    def equity(self, prices: Dict[str, float]) -> float:
        total = self.state.cash
        for symbol, pos in self.state.positions.items():
            px = prices.get(symbol, pos.avg_price)
            total += pos.quantity * px
        return total

    def get_positions(self) -> List[Position]:
        return list(self.state.positions.values())
