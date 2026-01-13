"""
Ingestion module - Knowledge graph extraction and PostgreSQL storage.
"""

from .async_ingestor import AsyncIngestor
from .postgres_storage import DirectPostgresStorage
from .pipeline import IngestionPipeline

__all__ = [
    "AsyncIngestor",
    "DirectPostgresStorage",
    "IngestionPipeline",
]
