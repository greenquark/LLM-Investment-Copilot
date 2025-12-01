"""Test script to verify data source configuration."""

import sys
from pathlib import Path
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.factory import create_data_engine_from_config

def test_backtest_config():
    """Test backtest configuration."""
    print("Testing backtest configuration...")
    config_file = project_root / "config" / "env.backtest.yaml"
    
    with open(config_file) as f:
        env = yaml.safe_load(f)
    
    print(f"✓ Config loaded from {config_file}")
    print(f"  Historical source: {env.get('data', {}).get('historical_source', 'N/A')}")
    print(f"  Real-time source: {env.get('data', {}).get('realtime_source', 'N/A')}")
    print(f"  Cache enabled: {env.get('data', {}).get('cache_enabled', 'N/A')}")
    print(f"  Available data sources: {list(env.get('data_sources', {}).keys())}")
    
    # Try to create data engine
    try:
        data_engine = create_data_engine_from_config(
            env_config=env,
            use_for="historical",
        )
        print(f"✓ Data engine created successfully: {type(data_engine).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to create data engine: {e}")
        return False

def test_live_config():
    """Test live trading configuration."""
    print("\nTesting live trading configuration...")
    config_file = project_root / "config" / "env.live.yaml"
    
    if not config_file.exists():
        print(f"⚠ Live config file not found: {config_file}")
        return True  # Not an error, just skip
    
    with open(config_file) as f:
        env = yaml.safe_load(f)
    
    print(f"✓ Config loaded from {config_file}")
    print(f"  Historical source: {env.get('data', {}).get('historical_source', 'N/A')}")
    print(f"  Real-time source: {env.get('data', {}).get('realtime_source', 'N/A')}")
    print(f"  Available data sources: {list(env.get('data_sources', {}).keys())}")
    
    # Try to create data engine for historical (realtime might need moomoo connection)
    try:
        data_engine = create_data_engine_from_config(
            env_config=env,
            use_for="historical",
        )
        print(f"✓ Historical data engine created successfully: {type(data_engine).__name__}")
        return True
    except Exception as e:
        print(f"✗ Failed to create historical data engine: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Data Source Configuration Test")
    print("=" * 60)
    
    success = True
    success &= test_backtest_config()
    success &= test_live_config()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

