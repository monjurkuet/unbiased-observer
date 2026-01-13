"""
Processing module - Community detection, summarization, and trigger management.
"""

from .coordinator import ProcessingCoordinator
from .trigger import ProcessingTrigger

__all__ = [
    "ProcessingCoordinator",
    "ProcessingTrigger",
]
