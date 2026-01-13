# Processing API

Community detection, graph analysis, and summarization components.

---

## Table of Contents

- [Overview](#overview)
- [ProcessingCoordinator](#processingcoordinator)
- [ProcessingTrigger](#processingtrigger)
- [Usage Examples](#usage-examples)

---

## Overview

The processing module analyzes the accumulated knowledge graph to identify research communities, detect patterns, and generate human-readable summaries using advanced graph algorithms and LLM-powered summarization.

### Key Components

- **ProcessingCoordinator**: Orchestrates community detection and summarization
- **ProcessingTrigger**: Determines when processing should run

---

## ProcessingCoordinator

Main coordinator for knowledge graph processing and analysis.

### Class Signature

```python
class ProcessingCoordinator:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def initialize(self) -> None
    async def run_processing_pipeline(self) -> Dict[str, Any]
    async def should_process(self) -> bool
```

### Methods

#### initialize()

Initialize processing components and dependencies.

```python
async def initialize(self) -> None:
    """Initialize community detector and summarizer."""
```

**Initialization**:
- Loads CommunityDetector from knowledge_base
- Loads CommunitySummarizer from knowledge_base
- Establishes database connections
- Prepares LLM client for summarization

#### run_processing_pipeline()

Execute complete processing pipeline.

```python
async def run_processing_pipeline(self) -> Dict[str, Any]:
    """Run community detection and summarization."""
```

**Returns**: Processing results

```python
{
    "status": "completed",
    "nodes_processed": 1250,
    "communities_created": 8,
    "summaries_generated": 8,
    "processing_time_seconds": 45.2,
    "algorithm": "leiden",
    "resolution": 1.0
}
```

**Pipeline Stages**:
1. **Graph Loading**: Load full knowledge graph from PostgreSQL
2. **Community Detection**: Apply Leiden algorithm to identify clusters
3. **Community Storage**: Save community assignments to database
4. **Summarization**: Generate LLM-powered community descriptions
5. **Metrics Collection**: Record processing statistics

#### should_process()

Check if processing should run based on triggers.

```python
async def should_process(self) -> bool:
    """Check if processing conditions are met."""
```

**Trigger Conditions**:
- Minimum entity count reached
- Sufficient time since last processing
- Manual trigger override

---

## ProcessingTrigger

Manages when knowledge graph processing should execute.

### Class Signature

```python
class ProcessingTrigger:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def should_trigger(self) -> bool
    async def _check_entity_count(self) -> bool
    async def _check_time_interval(self) -> bool
    async def record_processing_time(self) -> None
```

### Methods

#### should_trigger()

Determine if processing should run.

```python
async def should_trigger(self) -> bool:
    """Check all trigger conditions."""
```

**Returns**: True if processing should run

**Trigger Logic**:
- AND condition: All checks must pass
- Entity count check
- Time interval check
- Can be overridden for manual processing

#### _check_entity_count()

Check if minimum entities accumulated.

```python
async def _check_entity_count(self) -> bool:
    """Check if min entity threshold reached."""
```

**Configuration**:
```yaml
processing:
  min_entities_to_process: 100
```

#### _check_time_interval()

Check if enough time passed since last processing.

```python
async def _check_time_interval(self) -> bool:
    """Check time since last processing."""
```

**Configuration**:
```yaml
processing:
  min_time_between_processing_hours: 6
```

#### record_processing_time()

Record timestamp of processing execution.

```python
async def record_processing_time(self) -> None:
    """Record processing completion time."""
```

**Storage**: Updates agent_state table with last_processing_time

---

## Usage Examples

### Basic Processing Execution

```python
from research_agent.processing import ProcessingCoordinator

# Initialize coordinator
coordinator = ProcessingCoordinator(config)
await coordinator.initialize()

# Check if processing needed
if await coordinator.should_process():
    # Run processing pipeline
    result = await coordinator.run_processing_pipeline()

    print(f"Processed {result['nodes_processed']} nodes")
    print(f"Created {result['communities_created']} communities")
    print(f"Duration: {result['processing_time_seconds']}s")
else:
    print("Processing not needed yet")
```

### Trigger Management

```python
from research_agent.processing import ProcessingTrigger

trigger = ProcessingTrigger(config)

# Check trigger conditions
should_run = await trigger.should_trigger()
print(f"Should process: {should_run}")

# Manual trigger override
if manual_trigger:
    result = await coordinator.run_processing_pipeline()
    await trigger.record_processing_time()
```

### Integration with Task Queue

```python
from research_agent.orchestrator import TaskQueue
from research_agent.processing import ProcessingCoordinator

async def process_processing_task(task):
    coordinator = ProcessingCoordinator(config)
    await coordinator.initialize()

    # Run processing pipeline
    result = await coordinator.run_processing_pipeline()

    # Update task status
    await task_queue.update_task_status(
        task.id,
        "COMPLETED",
        result=result
    )

    print(f"Processing completed: {result}")
```

### Monitoring Processing Status

```python
# Check processing history
from research_agent.monitoring import MetricsCollector

metrics = MetricsCollector(config)
summary = await metrics.get_summary_metrics()

print(f"Last processing: {summary.get('last_processing_time')}")
print(f"Total communities: {summary.get('total_communities', 0)}")
print(f"Processing success rate: {summary.get('processing_success_rate', 0)}%")
```

---

## Configuration

### Processing Pipeline

```yaml
processing:
  enabled: true
  min_entities_to_process: 100
  min_time_between_processing_hours: 6

  community_detection:
    algorithm: "leiden"
    resolution: 1.0
    max_levels: 10

  summarization:
    enabled: true
    max_communities_per_batch: 10
```

### LLM Configuration for Summarization

```yaml
llm:
  model_pro: "gemini-2.5-pro"  # High-quality model for summarization
  max_retries: 3
  timeout: 120
```

### Database Configuration

```yaml
database:
  connection_string: "postgresql://agentzero@localhost:5432/knowledge_graph"
  pool_min_size: 5
  pool_max_size: 20
```

---

## Community Detection Algorithm

### Leiden Algorithm

The processing uses the Leiden algorithm for community detection:

**Algorithm Properties**:
- **Resolution**: Controls community granularity (higher = more communities)
- **Modularity Optimization**: Maximizes modularity score
- **Hierarchical**: Can detect nested community structures
- **Scalable**: Efficient for large graphs

**Parameters**:
- `resolution`: 1.0 (default) - balance between community size and count
- `max_levels`: 10 - maximum hierarchy depth

### Community Storage

Communities are stored in the database:

```sql
-- Community definitions
CREATE TABLE communities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(500),
    description TEXT,
    summary TEXT,
    size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Node-to-community mapping
CREATE TABLE community_membership (
    community_id INTEGER REFERENCES communities(id),
    node_id INTEGER REFERENCES nodes(id),
    importance FLOAT,
    PRIMARY KEY (community_id, node_id)
);
```

### Summarization Process

Each community gets an LLM-generated summary:

**Input**: Community member entities and relationships
**Output**: Human-readable description of research focus
**Model**: gemini-2.5-pro for high-quality summarization

**Example Summary**:
> "This community focuses on transformer architectures and self-attention mechanisms in natural language processing. Key contributions include the original transformer paper and subsequent improvements in efficiency and performance."

---

## Monitoring

### Processing Metrics

- **Nodes Processed**: Total entities analyzed
- **Communities Created**: Number of research communities identified
- **Processing Time**: Total pipeline execution time
- **Summarization Success**: Percentage of successful community summaries

### Performance Metrics

- **Algorithm Runtime**: Community detection execution time
- **Memory Usage**: Peak memory consumption during processing
- **Database Queries**: Graph loading and storage operations
- **LLM API Usage**: Summarization token consumption

### Quality Metrics

- **Community Coherence**: Average modularity score
- **Summary Quality**: LLM-generated summary relevance
- **Processing Frequency**: How often processing runs
- **Data Freshness**: Time since last processing

---

## Troubleshooting

### Processing Not Triggering

**Problem**: Processing never runs automatically

**Solutions**:
```python
# Check entity count
from research_agent.processing import ProcessingTrigger
trigger = ProcessingTrigger(config)
entity_check = await trigger._check_entity_count()
time_check = await trigger._check_time_interval()

print(f"Entity check: {entity_check}")
print(f"Time check: {time_check}")

# Check configuration
print(f"Min entities: {config.processing.min_entities_to_process}")
print(f"Min hours: {config.processing.min_time_between_processing_hours}")
```

### Community Detection Issues

**Problem**: Poor community quality or no communities found

**Solutions**:
```yaml
# Adjust resolution parameter
processing:
  community_detection:
    resolution: 0.8  # Lower = fewer, larger communities

# Check graph size
SELECT COUNT(*) FROM nodes;
SELECT COUNT(*) FROM edges;
```

### Summarization Failures

**Problem**: Community summaries not generated

**Solutions**:
```python
# Check LLM configuration
print(f"LLM model: {config.llm.model_pro}")
print(f"API key set: {bool(config.llm.api_key)}")

# Test LLM connectivity
test_summary = await summarizer.summarize_community(community_data)
print(f"Summarization working: {bool(test_summary)}")
```

### Performance Issues

**Problem**: Processing takes too long

**Solutions**:
```yaml
# Reduce community count
processing:
  community_detection:
    resolution: 1.2  # Higher = more, smaller communities

# Limit summarization batch
processing:
  summarization:
    max_communities_per_batch: 5
```

### Database Issues

**Problem**: Graph loading fails

**Solutions**:
```sql
-- Check data integrity
SELECT COUNT(*) FROM nodes;
SELECT COUNT(*) FROM edges;

-- Verify indexes
SELECT * FROM pg_indexes WHERE tablename IN ('nodes', 'edges');

-- Check for orphaned edges
SELECT COUNT(*) FROM edges e
LEFT JOIN nodes n1 ON e.source_id = n1.id
LEFT JOIN nodes n2 ON e.target_id = n2.id
WHERE n1.id IS NULL OR n2.id IS NULL;
```

---

## Performance Tuning

### Algorithm Optimization

- **Resolution Tuning**: Balance community count vs. size
- **Graph Preprocessing**: Remove isolated nodes
- **Incremental Processing**: Process only new data

### Database Optimization

- **Graph Loading**: Optimize large graph queries
- **Batch Storage**: Efficient community storage
- **Indexing**: Optimize community membership queries

### LLM Optimization

- **Batch Summarization**: Process multiple communities together
- **Prompt Engineering**: Optimize summarization prompts
- **Caching**: Cache summaries for unchanged communities

### Processing Scheduling

- **Trigger Optimization**: Balance frequency vs. overhead
- **Resource Limits**: Control memory and CPU usage
- **Parallel Processing**: Process independent communities concurrently

---

## Advanced Usage

### Custom Community Detection

```python
# Access underlying community detector
from knowledge_base.community import CommunityDetector

detector = CommunityDetector(config)
graph = await detector.load_graph()

# Custom algorithm parameters
communities = await detector.detect_communities(
    algorithm="leiden",
    resolution=1.5,
    random_state=42
)

print(f"Found {len(communities)} communities")
```

### Manual Summarization

```python
# Access summarizer directly
from knowledge_base.summarizer import CommunitySummarizer

summarizer = CommunitySummarizer(config)

# Summarize specific community
community_data = {
    "id": 123,
    "members": ["entity1", "entity2", "entity3"],
    "relationships": ["rel1", "rel2"]
}

summary = await summarizer.summarize_community(community_data)
print(f"Community summary: {summary}")
```

### Processing Analytics

```sql
-- Analyze processing results
SELECT
    c.id,
    c.name,
    c.size,
    c.summary,
    COUNT(cm.node_id) as member_count
FROM communities c
LEFT JOIN community_membership cm ON c.id = cm.community_id
GROUP BY c.id, c.name, c.size, c.summary
ORDER BY c.size DESC;
```

---

**Last Updated**: January 14, 2026