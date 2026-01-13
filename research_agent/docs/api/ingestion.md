# Ingestion API

Content processing pipeline, LLM-powered entity extraction, and PostgreSQL storage.

---

## Table of Contents

- [Overview](#overview)
- [IngestionPipeline](#ingestionpipeline)
- [AsyncIngestor](#asyncingestor)
- [DirectPostgresStorage](#directpostgresstorage)
- [Usage Examples](#usage-examples)

---

## Overview

The ingestion module transforms raw research content into structured knowledge graph data through LLM-powered entity extraction and efficient PostgreSQL storage with vector embeddings.

### Key Components

- **IngestionPipeline**: Orchestrates the complete ingestion workflow
- **AsyncIngestor**: LLM-powered entity and relationship extraction
- **DirectPostgresStorage**: PostgreSQL storage with vector embeddings

---

## IngestionPipeline

Main coordinator for the content ingestion workflow.

### Class Signature

```python
class IngestionPipeline:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def initialize(self) -> None
    async def ingest_content(self, content: str, metadata: Dict = None) -> Dict[str, Any]
    async def ingest_from_fetch_result(self, fetch_result: Dict) -> Dict[str, Any]
```

### Methods

#### initialize()

Initialize pipeline components and connections.

```python
async def initialize(self) -> None:
    """Initialize storage and ingestor components."""
```

**Initialization**:
- Creates DirectPostgresStorage instance
- Creates AsyncIngestor instance
- Establishes database connections
- Prepares LLM client

#### ingest_content()

Process raw content directly.

```python
async def ingest_content(
    self,
    content: str,
    metadata: Dict = None
) -> Dict[str, Any]:
    """Process content and store in knowledge graph."""
```

**Parameters**:
- `content`: Raw text content to process
- `metadata`: Optional metadata (source, author, etc.)

**Returns**: Processing results

```python
{
    "status": "completed",
    "entities_stored": 15,
    "relationships_stored": 23,
    "events_stored": 3,
    "processing_time_seconds": 12.5,
    "embedding_generation_time": 2.3
}
```

**Process Flow**:
1. Extract entities, relationships, events using AsyncIngestor
2. Generate vector embeddings for entities
3. Store entities in PostgreSQL with embeddings
4. Store relationships between entities
5. Store temporal events
6. Return processing statistics

#### ingest_from_fetch_result()

Process content from fetch operation result.

```python
async def ingest_from_fetch_result(self, fetch_result: Dict) -> Dict[str, Any]:
    """Process content from fetch result."""
```

**Parameters**:
- `fetch_result`: Result from ContentFetcher

**Fetch Result Format**:
```python
{
    "url": "https://arxiv.org/pdf/2301.07041.pdf",
    "content": "Raw PDF text content...",
    "content_type": "pdf",
    "metadata": {
        "source": "arxiv",
        "paper_id": "2301.07041",
        "title": "Attention Is All You Need"
    }
}
```

---

## AsyncIngestor

Asynchronous wrapper around GraphIngestor for LLM-powered extraction.

### Class Signature

```python
class AsyncIngestor:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def extract_async(self, content: str, metadata: Dict = None) -> Dict[str, Any]
    async def batch_extract_async(self, contents: List[str], metadata: List[Dict] = None) -> List[Dict[str, Any]]
```

### Methods

#### extract_async()

Extract entities and relationships from single content.

```python
async def extract_async(
    self,
    content: str,
    metadata: Dict = None
) -> Dict[str, Any]:
    """Extract knowledge from content asynchronously."""
```

**Parameters**:
- `content`: Text content to analyze
- `metadata`: Optional context information

**Returns**: Extracted knowledge structure

```python
{
    "entities": [
        {
            "name": "Transformer Architecture",
            "type": "concept",
            "description": "Neural network architecture using self-attention"
        },
        {
            "name": "Ashish Vaswani",
            "type": "person",
            "description": "Lead author of Attention Is All You Need"
        }
    ],
    "relationships": [
        {
            "source": "Ashish Vaswani",
            "target": "Transformer Architecture",
            "type": "authored",
            "description": "Vaswani co-authored the transformer paper"
        }
    ],
    "events": [
        {
            "primary_entity": "Transformer Architecture",
            "event_type": "publication",
            "event_date": "2017-06-12",
            "description": "Original transformer paper published"
        }
    ]
}
```

**LLM Processing**:
- **Core Pass**: Primary entity and relationship identification
- **Gleaning Pass**: Secondary relationship discovery
- **Structured Output**: Pydantic models ensure type safety
- **Error Handling**: Retry logic for API failures

#### batch_extract_async()

Extract from multiple contents concurrently.

```python
async def batch_extract_async(
    self,
    contents: List[str],
    metadata: List[Dict] = None
) -> List[Dict[str, Any]]:
    """Extract from multiple contents concurrently."""
```

**Parameters**:
- `contents`: List of text contents
- `metadata`: List of metadata dictionaries

**Concurrency Control**:
- Semaphore-based limiting (configurable max concurrent)
- Error isolation (one failure doesn't stop batch)
- Resource pooling for LLM API calls

---

## DirectPostgresStorage

PostgreSQL storage layer with vector embeddings and batch operations.

### Class Signature

```python
class DirectPostgresStorage:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def initialize(self) -> None
    async def close(self) -> None
    async def generate_embedding(self, text: str) -> List[float]
    async def store_entity(self, entity: Dict, embedding: List[float] = None) -> int
    async def store_entities_batch(self, entities: List[Dict]) -> List[int]
    async def store_edge(self, edge: Dict) -> int
    async def store_edges_batch(self, edges: List[Dict]) -> List[int]
    async def store_event(self, event: Dict) -> int
    async def store_events_batch(self, events: List[Dict]) -> List[int]
```

### Methods

#### initialize()

Initialize database connection pool.

```python
async def initialize(self) -> None:
    """Initialize PostgreSQL connection pool."""
```

**Connection Pooling**:
- Configurable min/max connections
- Async connection management
- Automatic reconnection on failures

#### close()

Close database connections.

```python
async def close(self) -> None:
    """Close connection pool."""
```

#### generate_embedding()

Generate vector embedding for text.

```python
async def generate_embedding(self, text: str) -> List[float]:
    """Generate vector embedding using Google GenAI."""
```

**Parameters**:
- `text`: Text to embed

**Returns**: 768-dimensional vector (for text-embedding-004)

**API Integration**:
- Uses Google GenAI embed_content API
- Batch processing for efficiency
- Error handling with retries

#### store_entity()

Store single entity with embedding.

```python
async def store_entity(
    self,
    entity: Dict,
    embedding: List[float] = None
) -> int:
    """Store entity in knowledge graph."""
```

**Parameters**:
- `entity`: Entity dictionary (name, type, description)
- `embedding`: Pre-computed embedding vector

**Returns**: Entity ID

**Conflict Resolution**:
- ON CONFLICT DO UPDATE for existing entities
- Preserves existing data, updates new fields

#### store_entities_batch()

Store multiple entities efficiently.

```python
async def store_entities_batch(self, entities: List[Dict]) -> List[int]:
    """Store multiple entities in batch."""
```

**Batch Processing**:
- Single transaction for all entities
- Bulk embedding generation
- Optimized INSERT statements
- Rollback on any failure

#### store_edge()

Store relationship between entities.

```python
async def store_edge(self, edge: Dict) -> int:
    """Store relationship between entities."""
```

**Parameters**:
- `edge`: Relationship dictionary

**Edge Structure**:
```python
{
    "source_id": 123,
    "target_id": 456,
    "type": "authored",
    "weight": 1.0,
    "description": "Author relationship"
}
```

**Uniqueness**: Composite unique constraint on (source_id, target_id, type)

#### store_edges_batch()

Store multiple relationships.

```python
async def store_edges_batch(self, edges: List[Dict]) -> List[int]:
    """Store multiple edges in batch."""
```

#### store_event()

Store temporal event.

```python
async def store_event(self, event: Dict) -> int:
    """Store temporal event."""
```

**Parameters**:
- `event`: Event dictionary

**Event Structure**:
```python
{
    "primary_entity_id": 123,
    "event_type": "publication",
    "event_date": "2017-06-12",
    "description": "Paper published"
}
```

#### store_events_batch()

Store multiple events.

```python
async def store_events_batch(self, events: List[Dict]) -> List[int]:
    """Store multiple events in batch."""
```

---

## Usage Examples

### Basic Content Ingestion

```python
from research_agent.ingestion import IngestionPipeline

# Initialize pipeline
pipeline = IngestionPipeline(config)
await pipeline.initialize()

# Ingest research paper content
content = """
Recent advances in transformer architectures have revolutionized
natural language processing. The key innovation is the multi-head
attention mechanism that allows the model to attend to different
parts of the input simultaneously.
"""

result = await pipeline.ingest_content(
    content=content,
    metadata={
        "source": "arxiv",
        "title": "Advances in Transformers",
        "authors": ["Jane Smith", "John Doe"]
    }
)

print(f"Stored {result['entities_stored']} entities")
print(f"Processing time: {result['processing_time_seconds']}s")
```

### Batch Processing

```python
from research_agent.ingestion import AsyncIngestor

ingestor = AsyncIngestor(config)

# Process multiple documents
contents = [
    "First research paper about machine learning...",
    "Second paper about neural networks...",
    "Third paper about computer vision..."
]

metadata = [
    {"title": "ML Paper 1", "year": 2023},
    {"title": "NN Paper 2", "year": 2023},
    {"title": "CV Paper 3", "year": 2023}
]

results = await ingestor.batch_extract_async(contents, metadata)

for i, result in enumerate(results):
    print(f"Document {i+1}: {len(result['entities'])} entities extracted")
```

### Direct Storage Operations

```python
from research_agent.ingestion import DirectPostgresStorage

storage = DirectPostgresStorage(config)
await storage.initialize()

# Store entity with embedding
entity = {
    "name": "Neural Networks",
    "type": "concept",
    "description": "Computational models inspired by biological neural networks"
}

entity_id = await storage.store_entity(entity)
print(f"Stored entity with ID: {entity_id}")

# Store relationship
edge = {
    "source_id": entity_id,
    "target_id": 456,  # Another entity ID
    "type": "related_to",
    "weight": 0.8,
    "description": "Neural networks are related to deep learning"
}

edge_id = await storage.store_edge(edge)
print(f"Stored relationship with ID: {edge_id}")

await storage.close()
```

### Integration with Task Queue

```python
from research_agent.orchestrator import TaskQueue
from research_agent.ingestion import IngestionPipeline

async def process_ingestion_task(task):
    pipeline = IngestionPipeline(config)
    await pipeline.initialize()

    # Extract content from task payload
    content = task.payload["content"]
    metadata = task.payload.get("metadata", {})

    # Process content
    result = await pipeline.ingest_content(content, metadata)

    # Update task status
    await task_queue.update_task_status(
        task.id,
        "COMPLETED",
        result=result
    )

    await pipeline.close()
```

### Embedding Generation

```python
from research_agent.ingestion import DirectPostgresStorage

storage = DirectPostgresStorage(config)
await storage.initialize()

# Generate embedding for text
text = "Transformer architecture uses self-attention mechanisms"
embedding = await storage.generate_embedding(text)

print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

await storage.close()
```

---

## Configuration

### Ingestion Pipeline

```yaml
ingestion:
  max_content_length: 100000  # characters
  max_retries: 3
  batch_size: 10
  concurrent_ingestions: 3

  content_types:
    - type: "pdf"
      max_size_mb: 50
      enabled: true
    - type: "html"
      enabled: true
    - type: "markdown"
      enabled: true
    - type: "text"
      enabled: true
```

### LLM Configuration

```yaml
llm:
  base_url: "http://localhost:8317/v1"
  api_key: "lm-studio"
  model_default: "gemini-2.5-flash"
  model_pro: "gemini-2.5-pro"
  max_retries: 3
  timeout: 120
```

### Embedding Configuration

```yaml
embedding:
  model: "text-embedding-004"
  dimension: 768
  batch_size: 100
```

### Database Configuration

```yaml
database:
  connection_string: "postgresql://agentzero@localhost:5432/knowledge_graph"
  pool_min_size: 5
  pool_max_size: 20
```

---

## Monitoring

### Ingestion Metrics

- **Content Processed**: Documents ingested per hour
- **Entity Extraction Rate**: Entities extracted per document
- **Relationship Density**: Average relationships per entity
- **Processing Latency**: Time from content to storage

### LLM Metrics

- **API Calls**: Total LLM API requests
- **Token Usage**: Input/output tokens consumed
- **Success Rate**: Successful extraction operations
- **Retry Rate**: Failed calls requiring retries

### Storage Metrics

- **Batch Efficiency**: Operations per batch
- **Embedding Generation**: Time per embedding
- **Database Latency**: Query execution times
- **Storage Growth**: Data volume over time

---

## Troubleshooting

### LLM Extraction Issues

**Problem**: Poor entity extraction quality

**Solutions**:
```python
# Check content length
if len(content) > 100000:
    print("Content too long, consider chunking")

# Verify LLM configuration
print(f"Model: {config.llm.model_default}")
print(f"API Key set: {bool(config.llm.api_key)}")

# Test LLM connectivity
test_result = await ingestor.extract_async("Test content")
print(f"LLM working: {len(test_result['entities'])} entities")
```

### Database Connection Issues

**Problem**: Connection pool exhausted

**Solutions**:
```yaml
# Increase pool size
database:
  pool_max_size: 30  # Increase from 20

# Check connection status
conn_count = await storage.check_connection()
print(f"Active connections: {conn_count}")
```

### Embedding Generation Issues

**Problem**: Embedding API failures

**Solutions**:
```python
# Check API quota
# Reduce batch size
embedding:
  batch_size: 50  # Reduce from 100

# Test embedding generation
test_embedding = await storage.generate_embedding("test text")
print(f"Embedding dimension: {len(test_embedding)}")
```

### Batch Processing Issues

**Problem**: Memory issues with large batches

**Solutions**:
```yaml
# Reduce batch sizes
ingestion:
  batch_size: 5  # Reduce from 10

# Monitor memory usage
import psutil
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")
```

### Content Processing Issues

**Problem**: Empty or malformed extraction results

**Solutions**:
```python
# Check content quality
print(f"Content length: {len(content)}")
print(f"Content preview: {content[:200]}...")

# Test with simple content
simple_result = await ingestor.extract_async("John Smith wrote a paper about AI.")
print(f"Simple extraction: {simple_result}")
```

---

## Performance Tuning

### LLM Optimization

- **Model Selection**: Use flash model for speed, pro for quality
- **Batch Processing**: Process multiple documents together
- **Content Chunking**: Split long documents
- **Caching**: Cache embeddings for repeated content

### Database Optimization

- **Connection Pooling**: Right-size pool for workload
- **Batch Inserts**: Use batch operations for efficiency
- **Indexing**: Optimize queries with proper indexes
- **Partitioning**: Consider time-based partitioning

### Processing Optimization

- **Concurrency Control**: Balance parallelism with resource limits
- **Memory Management**: Stream large content processing
- **Error Handling**: Fast-fail on persistent errors
- **Monitoring**: Track performance bottlenecks

---

**Last Updated**: January 14, 2026