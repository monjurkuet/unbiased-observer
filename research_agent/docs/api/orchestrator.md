# Orchestrator API

Task scheduling, queue management, and error recovery components.

---

## Table of Contents

- [Overview](#overview)
- [AgentScheduler](#agentscheduler)
- [TaskQueue](#taskqueue)
- [ErrorRecovery](#errorrecovery)
- [Usage Examples](#usage-examples)

---

## Overview

The orchestrator module manages the 24/7 operation of the Autonomous Research Agent through task scheduling, persistent queue management, and robust error recovery. It ensures continuous operation with automatic retries, health monitoring, and scalable task processing.

### Key Components

- **AgentScheduler**: APScheduler-based task coordination
- **TaskQueue**: PostgreSQL-backed persistent task storage
- **ErrorRecovery**: Exponential backoff retry mechanisms

---

## AgentScheduler

The main scheduler that coordinates all agent operations using APScheduler.

### Class Signature

```python
class AgentScheduler:
    def __init__(self, config: Config, logger: logging.Logger)
    async def start(self) -> None
    async def stop(self) -> None
```

### Methods

#### start()

Starts the scheduler and begins periodic task execution.

```python
async def start(self) -> None:
    """Initialize and start all scheduled jobs."""
```

**Jobs Started**:
- `process_queue()`: Every 10 seconds - Process pending tasks
- `process_ingestion_queue()`: Immediate - Handle ingestion tasks
- `retry_failed_tasks()`: Every 5 minutes - Retry failed tasks
- `monitoring()`: Every 5 minutes - Health checks
- `arxiv_monitoring()`: Every 2 hours - Research discovery

#### stop()

Gracefully stops the scheduler and cancels all running jobs.

```python
async def stop(self) -> None:
    """Stop scheduler and cleanup resources."""
```

### Internal Methods

#### _process_queue()

Processes the next available task from the queue.

```python
async def _process_queue(self) -> None:
    """Process next available task from queue."""
```

**Task Routing**:
- `FETCH` tasks → `_handle_fetch_task()`
- `INGEST` tasks → `_handle_ingest_task()`
- `PROCESS` tasks → `_handle_process_task()`

#### _handle_fetch_task()

Handles content fetching tasks.

```python
async def _handle_fetch_task(self, task: ResearchTask) -> None:
    """Handle FETCH task execution."""
```

**Process**:
1. Extract URL/file path from task payload
2. Call ContentFetcher to retrieve content
3. Call ContentExtractor to normalize text
4. Create INGEST task with extracted content
5. Update task status

#### _handle_ingest_task()

Handles content ingestion tasks.

```python
async def _handle_ingest_task(self, task: ResearchTask) -> None:
    """Handle INGEST task execution."""
```

**Process**:
1. Extract content from task payload
2. Call IngestionPipeline to process content
3. Store extracted entities in knowledge graph
4. Log ingestion metrics
5. Update task status

#### _handle_process_task()

Handles knowledge graph processing tasks.

```python
async def _handle_process_task(self, task: ResearchTask) -> None:
    """Handle PROCESS task execution."""
```

**Process**:
1. Check processing triggers
2. Call ProcessingCoordinator for community detection
3. Generate community summaries
4. Log processing metrics
5. Update task status

#### _retry_failed_tasks()

Retries tasks that have failed within the retry window.

```python
async def _retry_failed_tasks(self) -> None:
    """Retry failed tasks within retry window."""
```

**Logic**:
- Find FAILED tasks with retry_count < max_retries
- Reset status to PENDING
- Increment retry_count
- Apply exponential backoff delay

#### _monitoring()

Performs system health checks.

```python
async def _monitoring(self) -> None:
    """Perform system health monitoring."""
```

**Checks**:
- Database connectivity
- Task queue status
- LLM API availability
- Disk space and memory usage

#### _arxiv_monitoring()

Monitors arXiv for new research papers.

```python
async def _arxiv_monitoring(self) -> None:
    """Monitor arXiv for new research content."""
```

**Process**:
- Query configured arXiv categories/keywords
- Compare with existing sources
- Create FETCH tasks for new papers
- Update last monitoring timestamp

---

## TaskQueue

PostgreSQL-backed persistent task queue with concurrency control.

### Class Signature

```python
class TaskQueue:
    def __init__(self, config: Config, logger: logging.Logger)
    async def initialize(self) -> None
    async def add_task(self, task_type: str, payload: dict, priority: str = "medium") -> int
    async def get_next_task(self) -> Optional[ResearchTask]
    async def update_task_status(self, task_id: int, status: str, result: dict = None, error_message: str = None) -> None
    async def increment_retry(self, task_id: int) -> bool
    async def get_failed_tasks(self, max_age_hours: int = 1) -> List[ResearchTask]
    async def get_pending_count(self) -> int
    async def get_last_failed_tasks(self, limit: int = 10) -> List[ResearchTask]
```

### Methods

#### initialize()

Initialize database connection and prepare for operations.

```python
async def initialize(self) -> None:
    """Initialize database connection."""
```

#### add_task()

Add a new task to the queue.

```python
async def add_task(
    self,
    task_type: str,  # "FETCH", "INGEST", or "PROCESS"
    payload: dict,   # Task-specific data
    priority: str = "medium"  # "high", "medium", "low"
) -> int:
    """Add task to queue. Returns task ID."""
```

**Parameters**:
- `task_type`: Task type (FETCH, INGEST, PROCESS)
- `payload`: Task data (URL for FETCH, content for INGEST, etc.)
- `priority`: Task priority level

**Returns**: Task ID for tracking

#### get_next_task()

Retrieve next available task with concurrency safety.

```python
async def get_next_task(self) -> Optional[ResearchTask]:
    """Get next available task with row locking."""
```

**Concurrency Control**:
- Uses `FOR UPDATE SKIP LOCKED` to prevent duplicate processing
- Only returns PENDING tasks
- Updates status to IN_PROGRESS atomically

#### update_task_status()

Update task progress and completion status.

```python
async def update_task_status(
    self,
    task_id: int,
    status: str,  # "IN_PROGRESS", "COMPLETED", "FAILED"
    result: dict = None,
    error_message: str = None
) -> None:
    """Update task status and metadata."""
```

**Status Values**:
- `IN_PROGRESS`: Task started processing
- `COMPLETED`: Task finished successfully
- `FAILED`: Task failed with error

#### increment_retry()

Increment retry count for failed tasks.

```python
async def increment_retry(self, task_id: int) -> bool:
    """Increment retry count. Returns True if can retry."""
```

**Logic**:
- Increments retry_count
- Returns True if retry_count < max_retries
- Returns False if max_retries exceeded

#### get_failed_tasks()

Retrieve failed tasks within time window.

```python
async def get_failed_tasks(self, max_age_hours: int = 1) -> List[ResearchTask]:
    """Get failed tasks for retry consideration."""
```

#### get_pending_count()

Get count of pending tasks.

```python
async def get_pending_count(self) -> int:
    """Get number of pending tasks in queue."""
```

#### get_last_failed_tasks()

Get recent failed tasks for debugging.

```python
async def get_last_failed_tasks(self, limit: int = 10) -> List[ResearchTask]:
    """Get recent failed tasks for analysis."""
```

---

## ErrorRecovery

Exponential backoff retry mechanism with decorator pattern.

### Class Signature

```python
class ErrorRecovery:
    def __init__(self, config: Config, logger: logging.Logger)
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any
```

### Decorator

#### with_retry

Decorator for automatic retry with exponential backoff.

```python
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Decorator for retry logic with exponential backoff."""
```

**Parameters**:
- `max_retries`: Maximum number of retry attempts
- `base_delay`: Initial delay in seconds
- `backoff_factor`: Delay multiplier for each retry

**Usage**:
```python
@with_retry(max_retries=3, base_delay=1.0)
async def unreliable_operation():
    # Operation that might fail
    pass
```

### Methods

#### execute_with_retry()

Execute function with retry logic.

```python
async def execute_with_retry(
    self,
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Execute function with retry on failure."""
```

**Retry Logic**:
1. Execute function
2. On failure, wait with exponential backoff
3. Retry up to max_retries
4. Raise MaxRetriesExceeded on final failure

### Exceptions

#### MaxRetriesExceeded

Raised when maximum retry attempts are exceeded.

```python
class MaxRetriesExceeded(Exception):
    """Raised when max retries exceeded."""
    def __init__(self, operation: str, attempts: int, last_error: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
```

---

## Usage Examples

### Basic Task Queue Usage

```python
from research_agent.orchestrator import TaskQueue
from research_agent.config import load_config

# Initialize
config = load_config()
queue = TaskQueue(config, logger)
await queue.initialize()

# Add a fetch task
task_id = await queue.add_task(
    task_type="FETCH",
    payload={"url": "https://arxiv.org/pdf/2301.07041.pdf"},
    priority="high"
)

# Process tasks
while True:
    task = await queue.get_next_task()
    if task:
        # Process task
        await queue.update_task_status(task.id, "COMPLETED")
    else:
        break
```

### Scheduler Integration

```python
from research_agent.orchestrator import AgentScheduler

# Start autonomous operation
scheduler = AgentScheduler(config, logger)
await scheduler.start()

# Agent runs continuously, processing tasks every 10 seconds
# Monitoring every 5 minutes, arXiv checking every 2 hours

# Graceful shutdown
await scheduler.stop()
```

### Error Recovery Usage

```python
from research_agent.orchestrator import with_retry

@with_retry(max_retries=3, base_delay=1.0, backoff_factor=2.0)
async def fetch_content(url: str) -> str:
    """Fetch content with automatic retry on failure."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            return await response.text()

# Usage
try:
    content = await fetch_content("https://example.com/paper.pdf")
except MaxRetriesExceeded as e:
    logger.error(f"Failed after {e.attempts} attempts: {e.last_error}")
```

### Task Status Monitoring

```python
# Get queue statistics
pending_count = await queue.get_pending_count()
failed_tasks = await queue.get_last_failed_tasks(limit=5)

print(f"Pending tasks: {pending_count}")
for task in failed_tasks:
    print(f"Failed task {task.id}: {task.error_message}")
```

---

## Configuration

### Scheduler Configuration

```yaml
# In research_agent_config.yaml
orchestrator:
  queue_processing_interval: 10  # seconds
  retry_interval: 300            # 5 minutes
  monitoring_interval: 300       # 5 minutes
  arxiv_interval: 7200           # 2 hours
  max_concurrent_tasks: 5
```

### Task Queue Configuration

```yaml
# In research_agent_config.yaml
task_queue:
  max_retries: 3
  retry_window_hours: 1
  batch_size: 10
```

### Error Recovery Configuration

```yaml
# In research_agent_config.yaml
error_recovery:
  max_retries: 3
  base_delay: 1.0
  backoff_factor: 2.0
  max_delay: 60.0  # Maximum delay in seconds
```

---

## Monitoring

### Task Queue Metrics

- **Queue Depth**: Number of pending tasks
- **Processing Rate**: Tasks completed per minute
- **Error Rate**: Percentage of failed tasks
- **Retry Distribution**: Tasks by retry count

### Scheduler Metrics

- **Job Execution Times**: Duration of each scheduled job
- **Job Success Rates**: Success/failure rates per job type
- **System Health**: Database connectivity, API availability

### Error Recovery Metrics

- **Retry Attempts**: Total retry operations
- **Backoff Delays**: Average delay times
- **Final Failures**: Operations that exceeded max retries

---

## Troubleshooting

### Common Issues

#### Tasks Not Processing

**Symptoms**: Tasks remain in PENDING status

**Possible Causes**:
- Scheduler not running: Check `await scheduler.start()` called
- Database connection issues: Verify PostgreSQL connectivity
- Task queue locked: Check for long-running tasks

**Solutions**:
```bash
# Check scheduler status
ps aux | grep research_agent

# Verify database
psql -d knowledge_graph -c "SELECT status, COUNT(*) FROM research_tasks GROUP BY status;"

# Reset stuck tasks (CAUTION)
UPDATE research_tasks SET status = 'PENDING' WHERE status = 'IN_PROGRESS' AND updated_at < NOW() - INTERVAL '1 hour';
```

#### High Error Rates

**Symptoms**: Many tasks failing

**Possible Causes**:
- Network issues: Check internet connectivity
- API rate limits: Verify LLM API quotas
- Invalid content: Check source URLs/files

**Solutions**:
```bash
# Check recent failures
SELECT error_message, COUNT(*) FROM research_tasks
WHERE status = 'FAILED' AND updated_at > NOW() - INTERVAL '1 hour'
GROUP BY error_message ORDER BY count DESC;

# Adjust retry settings
UPDATE research_agent_config.yaml SET max_retries = 5;
```

#### Memory Issues

**Symptoms**: Out of memory errors

**Possible Causes**:
- Large content processing
- Concurrent task overload
- Memory leaks in LLM processing

**Solutions**:
```yaml
# Reduce concurrency
ingestion:
  concurrent_ingestions: 2

# Limit content size
research:
  max_content_length: 50000
```

---

## Performance Tuning

### Queue Optimization

- **Batch Processing**: Process multiple tasks together
- **Priority Queues**: Handle high-priority tasks first
- **Worker Pools**: Multiple concurrent task processors

### Database Optimization

- **Connection Pooling**: Reuse database connections
- **Indexing**: Optimize task queue queries
- **Partitioning**: Archive old completed tasks

### Retry Optimization

- **Smart Backoff**: Adaptive delay based on error type
- **Circuit Breakers**: Stop retrying consistently failing operations
- **Dead Letter Queues**: Move permanently failed tasks

---

**Last Updated**: January 14, 2026