"""
Test script to fetch and display CNN Fear & Greed Index.

Installation: pip install fear-and-greed
Documentation: https://pypi.org/project/fear-and-greed/
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path (if needed for any future imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import fear_and_greed
except ImportError:
    print("Error: 'fear-and-greed' package not found.")
    print("Please install it using: pip install fear-and-greed")
    sys.exit(1)


def format_index_description(description: str) -> str:
    """Format the index description for better readability."""
    # Capitalize first letter of each word
    return description.replace('_', ' ').title()


def get_fear_greed_color(value: float) -> str:
    """Return a color indicator based on the index value."""
    if value <= 25:
        return "🔴"  # Extreme Fear
    elif value <= 45:
        return "🟠"  # Fear
    elif value <= 55:
        return "🟡"  # Neutral
    elif value <= 75:
        return "🟢"  # Greed
    else:
        return "🟢🟢"  # Extreme Greed


def main():
    print("=" * 60)
    print("CNN Fear & Greed Index Test")
    print("=" * 60)
    print()
    
    try:
        # Fetch the Fear & Greed Index
        print("Fetching CNN Fear & Greed Index...")
        index_data = fear_and_greed.get()
        
        print()
        print("=" * 60)
        print("Current Fear & Greed Index")
        print("=" * 60)
        print()
        
        # Display the index value with color indicator
        color = get_fear_greed_color(index_data.value)
        print(f"Index Value: {color} {index_data.value:.2f} / 100")
        print()
        
        # Display description
        formatted_desc = format_index_description(index_data.description)
        print(f"Category: {formatted_desc}")
        print()
        
        # Display last update timestamp
        if index_data.last_update:
            # Convert to local time if timezone-aware
            if index_data.last_update.tzinfo:
                local_time = index_data.last_update.astimezone()
            else:
                local_time = index_data.last_update
            
            print(f"Last Updated: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"  (UTC: {index_data.last_update.strftime('%Y-%m-%d %H:%M:%S %Z')})")
        else:
            print("Last Updated: N/A")
        
        print()
        print("=" * 60)
        print("Interpretation:")
        print("=" * 60)
        print("0-25:   Extreme Fear (🔴)")
        print("26-45:  Fear (🟠)")
        print("46-55:  Neutral (🟡)")
        print("56-75:  Greed (🟢)")
        print("76-100: Extreme Greed (🟢🟢)")
        print()
        
        # Additional analysis
        print("=" * 60)
        print("Trading Implications:")
        print("=" * 60)
        if index_data.value <= 25:
            print("⚠️  Extreme Fear: Potential buying opportunity")
            print("   Markets may be oversold. Consider contrarian positions.")
        elif index_data.value <= 45:
            print("⚠️  Fear: Cautious sentiment")
            print("   Markets showing fear. Monitor for entry opportunities.")
        elif index_data.value <= 55:
            print("ℹ️  Neutral: Balanced sentiment")
            print("   Markets in equilibrium. No strong directional bias.")
        elif index_data.value <= 75:
            print("⚠️  Greed: Elevated sentiment")
            print("   Markets showing greed. Consider taking profits.")
        else:
            print("⚠️  Extreme Greed: Potential selling opportunity")
            print("   Markets may be overbought. Consider defensive positions.")
        
        print()
        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("Error fetching Fear & Greed Index")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Possible causes:")
        print("  - Network connectivity issues")
        print("  - CNN website temporarily unavailable")
        print("  - Rate limiting (requests are cached for 1 minute)")
        print()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
