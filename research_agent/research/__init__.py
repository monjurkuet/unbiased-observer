"""
Research module - Content fetching, extraction, and source management.
"""

from .content_fetcher import ContentFetcher, FetchError
from .content_extractor import ContentExtractor
from .source_discovery import SourceDiscovery
from .manual_source import ManualSourceManager

__all__ = [
    "ContentFetcher",
    "FetchError",
    "ContentExtractor",
    "SourceDiscovery",
    "ManualSourceManager",
]

from .arxiv_integrator import ArxivIntegrator, ArxivSourceManager

__all__.extend([
    "ArxivIntegrator",
    "ArxivSourceManager",
])
