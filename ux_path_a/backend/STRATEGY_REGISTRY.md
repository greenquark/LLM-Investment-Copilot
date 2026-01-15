# Strategy Registry Implementation

## Overview

Implemented a dynamic strategy registry system to locate and import strategies from `core/strategy/` without hardcoding imports. This solves the Railway deployment issue where direct imports of `core.strategy.llm_trend_detection` were failing.

## Implementation

### 1. Strategy Registry Module (`core/strategy/registry.py`)

The registry provides:
- **Dynamic discovery**: Automatically finds all Strategy classes in `core/strategy/`
- **Config discovery**: Also discovers Config classes (e.g., `LLMTrendDetectionConfig`)
- **Multiple fallback methods**: Tries `/app` (Railway), calculated path, and current working directory
- **Caching**: Registry is cached after first discovery for performance

### 2. Key Functions

- `get_strategy_class(strategy_name, project_root)`: Get a strategy class by name
- `get_config_class(config_name, project_root)`: Get a config class by name
- `list_strategies(project_root)`: List all available strategies
- `get_project_root()`: Auto-detect project root with multiple fallbacks

### 3. Updated `analysis_tools.py`

Now uses the registry instead of direct imports:
```python
from core.strategy.registry import get_strategy_class, get_config_class

LLMTrendDetectionStrategy = get_strategy_class("LLMTrendDetectionStrategy", project_root)
LLMTrendDetectionConfig = get_config_class("LLMTrendDetectionConfig", project_root)
```

**Fallback mechanism**: If registry fails, falls back to direct import for backwards compatibility.

## Benefits

1. **Extensibility**: New strategies are automatically discovered - no code changes needed
2. **Robustness**: Multiple fallback methods for finding project root
3. **Railway compatibility**: Works with `/app` path in Docker/Railway
4. **Backwards compatible**: Falls back to direct import if registry fails
5. **Maintainability**: No hardcoded strategy imports

## How It Works

1. Registry scans `core/strategy/*.py` files (excluding `base.py`, `strategy_utils.py`, etc.)
2. Uses `inspect` to find classes that inherit from `Strategy`
3. Caches results for performance
4. Provides easy lookup by class name

## Usage Example

```python
from core.strategy.registry import get_strategy_class, get_config_class

# Get strategy class
StrategyClass = get_strategy_class("LLMTrendDetectionStrategy")

# Get config class
ConfigClass = get_config_class("LLMTrendDetectionConfig")

# List all available strategies
from core.strategy.registry import list_strategies
all_strategies = list_strategies()
```

## Railway Deployment

The registry automatically detects Railway's `/app` path:
1. First checks if `/app/core/strategy` exists (Railway/Docker)
2. Falls back to calculated path from file location
3. Falls back to current working directory

This ensures it works in both local development and Railway deployment.
