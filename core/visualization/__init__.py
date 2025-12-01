from core.visualization.models import TradeSignal, IndicatorData

# Plotly chart visualizer (recommended)
from core.visualization.plotly_chart import PlotlyChartVisualizer, get_theme, THEMES

# FastAPI integration (optional)
def get_fastapi_app():
    """Lazy import of FastAPI app to avoid dependency at import time."""
    from core.visualization.fastapi_chart import create_app
    return create_app()

# Legacy support - Flask-based web chart
def get_web_chart_server():
    """Lazy import of WebChartServer to avoid Flask dependency."""
    from core.visualization.web_chart import WebChartServer
    return WebChartServer

# Legacy support - Matplotlib local chart
from core.visualization.local_chart import LocalChartVisualizer

__all__ = [
    "TradeSignal",
    "IndicatorData",
    "PlotlyChartVisualizer",
    "get_theme",
    "THEMES",
    "get_fastapi_app",
    "get_web_chart_server",  # Legacy
    "LocalChartVisualizer",  # Legacy
]

