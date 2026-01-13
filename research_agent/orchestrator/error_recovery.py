"""
Error Recovery - Exponential backoff retry mechanism
"""

import asyncio
import logging
from typing import Callable, Optional
from functools import wraps

logger = logging.getLogger('research_agent.orchestrator')


class MaxRetriesExceeded(Exception):
    """Maximum retries exceeded"""
    pass


class ErrorRecovery:
    """Error recovery with exponential backoff"""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        base_delay: float = 1.0
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.base_delay = base_delay

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_types: tuple = (Exception,),
        **kwargs
    ):
        """
        Execute function with exponential backoff retry

        Args:
            func: Async function to execute
            error_types: Tuple of exception types to catch
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function

        Raises:
            MaxRetriesExceeded: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)

            except error_types as e:
                last_exception = e

                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")
                    raise MaxRetriesExceeded(f"Function failed after {self.max_retries} retries") from e

                # Calculate backoff delay
                delay = self.base_delay * (self.backoff_factor ** attempt)

                logger.warning(
                    f"Attempt {attempt+1}/{self.max_retries} failed: {e}. Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        # Should not reach here
        raise MaxRetriesExceeded("Unexpected error recovery state")


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 1.0,
    error_types: tuple = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff

    Usage:
        @with_retry(max_retries=3)
        async def my_function():
            # Your code here
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            recovery = ErrorRecovery(max_retries, backoff_factor, base_delay)
            return await recovery.execute_with_retry(
                func,
                *args,
                error_types=error_types,
                **kwargs
            )
        return wrapper
    return decorator
