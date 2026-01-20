"""
Guardrails and safety controls for UX Path A.

Implements cost controls, safety warnings, and feature gating
as required by platform invariants.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import logging

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.models import TokenBudget, ChatSession
from ux_path_a.backend.backend_core.config import settings

logger = logging.getLogger(__name__)


class TokenBudgetTracker:
    """
    Tracks token usage per session (INV-LLM-03).
    
    Enforces per-session token budgets to control costs.
    """
    
    def __init__(self, db: Session):
        """Initialize tracker with database session."""
        self.db = db
    
    def check_budget(
        self,
        session_id: str,
        tokens_to_use: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if session has enough token budget.
        
        Args:
            session_id: Session identifier
            tokens_to_use: Tokens that will be used
            
        Returns:
            (allowed, error_message)
        """
        budget = self.db.query(TokenBudget).filter(
            TokenBudget.session_id == session_id
        ).first()
        
        if not budget:
            # Create budget if doesn't exist
            budget = TokenBudget(
                session_id=session_id,
                budget_limit=settings.MAX_TOKENS_PER_SESSION,
                tokens_used=0,
            )
            self.db.add(budget)
            self.db.commit()
        
        if budget.tokens_used + tokens_to_use > budget.budget_limit:
            error_msg = (
                f"Token budget exceeded. Used: {budget.tokens_used}, "
                f"Limit: {budget.budget_limit}, Requested: {tokens_to_use}"
            )
            logger.warning(f"Token budget exceeded for session {session_id}")
            return False, error_msg
        
        return True, None
    
    def record_usage(
        self,
        session_id: str,
        tokens_used: int,
    ):
        """Record token usage for a session."""
        budget = self.db.query(TokenBudget).filter(
            TokenBudget.session_id == session_id
        ).first()
        
        if budget:
            budget.tokens_used += tokens_used
            budget.updated_at = datetime.utcnow()
            self.db.commit()
        else:
            # Create if doesn't exist
            budget = TokenBudget(
                session_id=session_id,
                budget_limit=settings.MAX_TOKENS_PER_SESSION,
                tokens_used=tokens_used,
            )
            self.db.add(budget)
            self.db.commit()
        
        # Also update session
        session = self.db.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()
        if session:
            session.total_tokens_used += tokens_used
            self.db.commit()


class SafetyControls:
    """
    Safety controls and warnings (INV-SAFE-02, INV-SAFE-03).
    
    Provides warnings for high-volatility, leverage, and risk disclosures.
    """
    
    @staticmethod
    def check_volatility_warning(symbol: str, volatility: float) -> Optional[str]:
        """
        Check if volatility warning is needed.
        
        Args:
            symbol: Stock symbol
            volatility: Volatility percentage (annualized)
            
        Returns:
            Warning message if needed, None otherwise
        """
        if volatility > 50:  # High volatility threshold
            return (
                f"⚠️ HIGH VOLATILITY WARNING: {symbol} has high volatility ({volatility:.1f}%). "
                "This instrument carries significant risk. Past performance does not guarantee future results."
            )
        return None
    
    @staticmethod
    def check_leverage_warning(symbol: str) -> Optional[str]:
        """
        Check if leverage warning is needed.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Warning message if needed, None otherwise
        """
        # Check for leveraged ETFs (common patterns)
        leverage_keywords = ["TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "LABU", "LABD"]
        
        if any(keyword in symbol.upper() for keyword in leverage_keywords):
            return (
                f"⚠️ LEVERAGE WARNING: {symbol} is a leveraged ETF. "
                "Leveraged ETFs carry amplified risk and may experience significant losses. "
                "Not suitable for all investors."
            )
        return None
    
    @staticmethod
    def get_risk_disclosure() -> str:
        """
        Get standard risk disclosure (INV-SAFE-03).
        
        Returns:
            Risk disclosure text
        """
        return (
            "**RISK DISCLOSURE**: All trading decisions carry risk. Past performance does not "
            "guarantee future results. Consider consulting a qualified professional."
        )


class FeatureGate:
    """
    Feature gating system (INV-CROSS-01).
    
    Controls which features are available based on configuration.
    """
    
    def __init__(self):
        """Initialize feature gates."""
        # Feature flags (can be loaded from config/database)
        self._gates = {
            "backtesting": True,
            "portfolio_import": False,  # MVP+ feature
            "advanced_analysis": True,
        }
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self._gates.get(feature, False)
    
    def require_feature(self, feature: str) -> None:
        """Require a feature to be enabled, raise if not."""
        if not self.is_enabled(feature):
            raise ValueError(f"Feature '{feature}' is not enabled")


# Global feature gate instance
feature_gate = FeatureGate()
