"""
Error handling utilities for API calls and data operations.

This module provides common error handling patterns used across the codebase,
particularly for API rate limits, authentication errors, and network issues.
"""

from __future__ import annotations
from typing import List, Optional


# Common API error keywords that indicate rate limits or authentication issues
API_ERROR_KEYWORDS = [
    'rate limit',
    '429',
    'too many requests',
    'limit exceeded',
    'quota',
    '403',
    '401',
    'payment required',
    '402',
    'timeout',
    'unauthorized',
    'forbidden',
]


def is_api_error(exception: Exception) -> bool:
    """
    Check if an exception represents an API error that should be re-raised
    (rate limits, authentication issues, etc.) rather than being handled silently.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if this is an API error that should be re-raised, False otherwise
        
    Examples:
        >>> try:
        ...     # API call
        ... except Exception as e:
        ...     if is_api_error(e):
        ...         raise  # Re-raise API errors
        ...     else:
        ...         # Handle other errors
        ...         pass
    """
    error_str = str(exception).lower()
    return any(keyword in error_str for keyword in API_ERROR_KEYWORDS)


def format_api_error_message(
    source: str,
    symbol: Optional[str] = None,
    date: Optional[str] = None,
    error: Optional[Exception] = None,
    additional_info: Optional[str] = None,
) -> str:
    """
    Format a standardized API error message.
    
    Args:
        source: Data source name (e.g., "MarketData.app", "YFinance")
        symbol: Optional symbol that was being fetched
        date: Optional date that was being fetched
        error: Optional exception that occurred
        additional_info: Optional additional information to include
        
    Returns:
        Formatted error message string
        
    Examples:
        >>> msg = format_api_error_message("MarketData.app", symbol="AAPL", error=e)
        >>> logger.error(msg)
    """
    parts = [f"[{source}]"]
    
    if symbol:
        parts.append(f"symbol={symbol}")
    if date:
        parts.append(f"date={date}")
    
    if error:
        parts.append(f"error: {error}")
    
    if additional_info:
        parts.append(additional_info)
    
    return " ".join(parts)


def should_retry_error(exception: Exception, max_retries: int = 3) -> bool:
    """
    Determine if an error should trigger a retry.
    
    Args:
        exception: Exception that occurred
        max_retries: Maximum number of retries allowed
        
    Returns:
        True if the error should be retried, False otherwise
        
    Note:
        Rate limit errors (429) and timeouts are typically retryable,
        while authentication errors (401, 403) should not be retried.
    """
    error_str = str(exception).lower()
    
    # Don't retry authentication errors
    if any(keyword in error_str for keyword in ['401', '403', 'unauthorized', 'forbidden', 'payment required', '402']):
        return False
    
    # Retry rate limits and timeouts
    if any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests', 'timeout']):
        return True
    
    # For other errors, don't retry by default
    return False
