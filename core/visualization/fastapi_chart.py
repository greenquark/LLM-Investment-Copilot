"""
FastAPI integration for Plotly chart visualization.

Provides HTTP endpoints for serving interactive trading charts.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from core.models.bar import Bar
from core.visualization.models import TradeSignal, IndicatorData
from core.visualization.plotly_chart import PlotlyChartVisualizer, get_theme

# Global app instance (can be initialized elsewhere)
app: Optional[FastAPI] = None
_chart_data: Dict[str, dict] = {}


def create_app() -> FastAPI:
    """Create and configure FastAPI app for chart serving."""
    global app
    app = FastAPI(title="Trading Chart Server", version="1.0.0")
    _setup_routes()
    return app


def _setup_routes():
    """Setup FastAPI routes."""
    if app is None:
        return
    
    @app.get("/chart", response_class=HTMLResponse)
    async def get_chart(
        chart_id: str = Query(..., description="Chart identifier"),
        theme: str = Query("tradingview", description="Theme: moomoo, tradingview, dark, light"),
    ):
        """
        Get interactive chart HTML.
        
        Args:
            chart_id: Unique identifier for the chart data
            theme: Chart theme name
        """
        if chart_id not in _chart_data:
            return HTMLResponse(
                f"<h2>Chart not found: {chart_id}</h2>",
                status_code=404,
            )
        
        data = _chart_data[chart_id]
        visualizer = PlotlyChartVisualizer(theme=theme)
        
        try:
            visualizer.build_chart(
                bars=data["bars"],
                signals=data.get("signals"),
                indicator_data=data.get("indicator_data"),
                equity_curve=data.get("equity_curve"),
                metrics=data.get("metrics"),
                symbol=data.get("symbol", "UNKNOWN"),
                show_equity=data.get("show_equity", True),
            )
            html = visualizer.to_html()
            return HTMLResponse(content=html)
        except Exception as e:
            return HTMLResponse(
                f"<h2>Error building chart: {str(e)}</h2>",
                status_code=500,
            )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "charts": len(_chart_data)}


def register_chart(
    chart_id: str,
    bars: List[Bar],
    signals: Optional[List[TradeSignal]] = None,
    indicator_data: Optional[List[IndicatorData]] = None,
    equity_curve: Optional[Dict[datetime, float]] = None,
    metrics: Optional[Dict] = None,
    symbol: str = "UNKNOWN",
    show_equity: bool = True,
):
    """
    Register chart data for serving via HTTP.
    
    Args:
        chart_id: Unique identifier for this chart
        bars: List of price bars
        signals: Optional trading signals
        indicator_data: Optional indicator data
        equity_curve: Optional equity curve
        metrics: Optional performance metrics
        symbol: Stock symbol
        show_equity: Whether to show equity curve
    """
    _chart_data[chart_id] = {
        "bars": bars,
        "signals": signals,
        "indicator_data": indicator_data,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "symbol": symbol,
        "show_equity": show_equity,
    }


def clear_chart(chart_id: str):
    """Remove chart data."""
    if chart_id in _chart_data:
        del _chart_data[chart_id]

