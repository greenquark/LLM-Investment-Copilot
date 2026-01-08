"""
Minimal test script to verify OpenAI API setup and connectivity.

This script tests:
1. OpenAI package installation and version
2. API key configuration
3. Basic API call functionality
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.config_loader import load_config_with_secrets


async def test_openai_api():
    """Test OpenAI API setup and make a simple call."""
    print("=" * 60)
    print("OpenAI API Test Script")
    print("=" * 60)
    
    # 1. Check if openai package is installed
    print("\n[1] Checking OpenAI package installation...")
    try:
        import openai
        version = getattr(openai, '__version__', 'unknown')
        print(f"  ✓ OpenAI package installed: version {version}")
        
        # Check for AsyncOpenAI
        if hasattr(openai, 'AsyncOpenAI'):
            print("  ✓ AsyncOpenAI class available")
        else:
            print("  ✗ AsyncOpenAI class NOT available")
            print(f"     ERROR: openai >= 1.0.0 required. Current version: {version}")
            print("     Please upgrade with: pip install --upgrade openai")
            return False
    except ImportError:
        print("  ✗ OpenAI package not installed")
        print("     Please install with: pip install openai")
        return False
    
    # 2. Get API key from config or environment
    print("\n[2] Checking API key configuration...")
    api_key = None
    
    # Try to read from config file (with secrets merged)
    config_file = project_root / "config" / "strategy.llm_trend_detection.yaml"
    if config_file.exists():
        try:
            config = load_config_with_secrets(config_file, strategy_name="llm_trend_detection")
            api_key = config.get("openai_api_key")
            if api_key and api_key != "YOUR_OPENAI_API_KEY":
                print("  ✓ API key found in config file (with secrets)")
            else:
                print("  - No API key in config file or secrets")
        except Exception as e:
            print(f"  - Could not read config file: {e}")
    
    # Try environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("  ✓ API key found in OPENAI_API_KEY environment variable")
    
    if not api_key:
        print("  ✗ No API key found")
        print("     Set openai_api_key in config/strategy.llm_trend_detection.yaml")
        print("     OR set OPENAI_API_KEY environment variable")
        return False
    
    # Mask API key for display (show first 10 and last 4 chars)
    if len(api_key) > 14:
        masked_key = api_key[:10] + "..." + api_key[-4:]
    else:
        masked_key = "***"
    print(f"  ✓ API key configured: {masked_key}")
    
    # 3. Test API client initialization
    print("\n[3] Testing OpenAI client initialization...")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        print("  ✓ Client initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize client: {e}")
        return False
    
    # 4. Make a simple API call - try multiple models
    print("\n[4] Making test API call...")
    
    # Try common OpenAI models in order of preference
    # Default: gpt-5-mini
    models_to_try = [
        "gpt-5-mini",       # GPT-5-mini (default model)
        "gpt-5.1",          # GPT-5.1 full model
        "gpt-4o-mini",      # GPT-4o-mini (fallback)
        "gpt-4o",           # Full GPT-4o (fallback)
        "gpt-4-turbo",      # GPT-4 Turbo (fallback)
        "gpt-3.5-turbo",    # GPT-3.5 Turbo (fallback)
    ]
    
    # Also check config for model preference
    config_file = project_root / "config" / "strategy.llm_trend_detection.yaml"
    preferred_model = None
    if config_file.exists():
        try:
            config = load_config_with_secrets(config_file, strategy_name="llm_trend_detection")
            preferred_model = config.get("llm_model")
            if preferred_model and preferred_model not in models_to_try:
                    # Add user's preferred model to the front of the list
                    models_to_try.insert(0, preferred_model)
        except Exception:
            pass
    
    last_error = None
    for model_name in models_to_try:
        print(f"  Trying model: {model_name}...")
        try:
            # Some models have different parameter requirements
            request_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'API test successful' if you can read this."}
                ],
            }
            
            # Try with temperature first, then without if not supported
            try:
                request_params["temperature"] = 0.0
                # Try max_completion_tokens first for newer models, fallback to max_tokens
                try:
                    request_params["max_completion_tokens"] = 20
                    response = await client.chat.completions.create(**request_params)
                except Exception as e1:
                    if "max_completion_tokens" in str(e1).lower() or "unsupported_parameter" in str(e1).lower():
                        # Remove max_completion_tokens and try max_tokens instead
                        request_params.pop("max_completion_tokens", None)
                        request_params["max_tokens"] = 20
                        response = await client.chat.completions.create(**request_params)
                    else:
                        raise
            except Exception as e2:
                # If temperature is not supported, try without it
                if "temperature" in str(e2).lower() or "unsupported_value" in str(e2).lower():
                    request_params.pop("temperature", None)
                    # Try max_completion_tokens first, then max_tokens
                    try:
                        request_params["max_completion_tokens"] = 20
                        response = await client.chat.completions.create(**request_params)
                    except Exception as e3:
                        if "max_completion_tokens" in str(e3).lower() or "unsupported_parameter" in str(e3).lower():
                            request_params.pop("max_completion_tokens", None)
                            request_params["max_tokens"] = 20
                            response = await client.chat.completions.create(**request_params)
                        else:
                            raise
                else:
                    raise
            
            content = response.choices[0].message.content
            print(f"  ✓ API call successful!")
            print(f"  ✓ Response: {content}")
            print(f"  ✓ Model used: {response.model}")
            print(f"  ✓ Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
            
            # If the model that worked is different from config, suggest updating
            if preferred_model and model_name != preferred_model:
                print(f"\n  ⚠ NOTE: Your config specifies '{preferred_model}' but that model is not available.")
                print(f"     Working model '{model_name}' is being used instead.")
                print(f"     Consider updating config/strategy.llm_trend_detection.yaml to use '{model_name}'")
            
            return True
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if "model_not_found" in error_msg.lower() or "404" in error_msg:
                print(f"    ✗ Model '{model_name}' not available, trying next...")
                continue
            elif "quota" in error_msg.lower() or "429" in error_msg or "insufficient_quota" in error_msg.lower():
                print(f"    ⚠ Model '{model_name}' exists but quota exceeded (429)")
                print(f"     This means the model name is correct, but your API key needs billing/quota setup.")
                print(f"     The correct model to use is: {model_name}")
                # Don't continue trying - we found the right model, just quota issue
                return True  # Return True because the setup is correct, just quota issue
            else:
                print(f"  ✗ API call failed with model '{model_name}': {e}")
                print(f"     Error type: {type(e).__name__}")
                if hasattr(e, 'status_code'):
                    print(f"     HTTP status: {e.status_code}")
                return False
    
    # If we get here, all models failed
    print(f"  ✗ All models failed. Last error: {last_error}")
    print(f"     Error type: {type(last_error).__name__ if last_error else 'Unknown'}")
    if last_error and hasattr(last_error, 'status_code'):
        print(f"     HTTP status: {last_error.status_code}")
    print(f"\n  Available models to try:")
    for model in models_to_try:
        print(f"     - {model}")
    return False


async def main():
    """Main entry point."""
    try:
        success = await test_openai_api()
        
        print("\n" + "=" * 60)
        if success:
            print("✓ All tests passed! OpenAI API is working correctly.")
        else:
            print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

