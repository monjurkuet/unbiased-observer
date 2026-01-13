# Monitoring API

Health checks, metrics collection, and structured logging components.

---

## Table of Contents

- [Overview](#overview)
- [MetricsCollector](#metricscollector)
- [HealthChecker](#healthchecker)
- [Logging Setup](#logging-setup)
- [Usage Examples](#usage-examples)

---

## Overview

The monitoring module provides comprehensive observability for the Autonomous Research Agent through structured logging, performance metrics collection, and system health monitoring.

### Key Components

- **MetricsCollector**: Performance metrics and event tracking
- **HealthChecker**: System health validation (placeholder implementation)
- **setup_logging()**: Structured logging configuration

---

## MetricsCollector

Core metrics collection and performance monitoring.

### Class Signature

```python
class MetricsCollector:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def record_task_start(self, task_id: int, task_type: str) -> None
    async def record_task_complete(self, task_id: int, duration_seconds: float, success: bool) -> None
    async def record_task_failure(self, task_id: int, error_message: str) -> None
    async def record_ingestion_start(self, content_length: int) -> None
    async def record_ingestion_complete(self, entities_count: int, relationships_count: int, duration_seconds: float) -> None
    async def record_processing_start(self) -> None
    async def record_processing_complete(self, nodes_processed: int, communities_created: int, duration_seconds: float) -> None
    async def get_summary_metrics(self) -> Dict[str, Any]
```

### Methods

#### record_task_start()

Record task execution start.

```python
async def record_task_start(self, task_id: int, task_type: str) -> None:
    """Record task execution start."""
```

**Parameters**:
- `task_id`: Unique task identifier
- `task_type`: Task type (FETCH, INGEST, PROCESS)

#### record_task_complete()

Record successful task completion.

```python
async def record_task_complete(
    self,
    task_id: int,
    duration_seconds: float,
    success: bool
) -> None:
    """Record task completion with metrics."""
```

**Parameters**:
- `task_id`: Task identifier
- `duration_seconds`: Execution duration
- `success`: Completion status

#### record_task_failure()

Record task failure with error details.

```python
async def record_task_failure(self, task_id: int, error_message: str) -> None:
    """Record task failure with error."""
```

#### record_ingestion_start()

Record ingestion pipeline start.

```python
async def record_ingestion_start(self, content_length: int) -> None:
    """Record ingestion start."""
```

#### record_ingestion_complete()

Record ingestion completion with statistics.

```python
async def record_ingestion_complete(
    self,
    entities_count: int,
    relationships_count: int,
    duration_seconds: float
) -> None:
    """Record ingestion completion."""
```

#### record_processing_start()

Record processing pipeline start.

```python
async def record_processing_start(self) -> None:
    """Record processing start."""
```

#### record_processing_complete()

Record processing completion with statistics.

```python
async def record_processing_complete(
    self,
    nodes_processed: int,
    communities_created: int,
    duration_seconds: float
) -> None:
    """Record processing completion."""
```

#### get_summary_metrics()

Get comprehensive system metrics summary.

```python
async def get_summary_metrics(self) -> Dict[str, Any]:
    """Get summary of system metrics."""
```

**Returns**: Comprehensive metrics dictionary

```python
{
    "task_metrics": {
        "total_tasks": 1250,
        "completed_tasks": 1180,
        "failed_tasks": 45,
        "pending_tasks": 25,
        "success_rate": 94.4,
        "average_duration": 12.5
    },
    "ingestion_metrics": {
        "total_ingestions": 890,
        "entities_extracted": 15420,
        "relationships_extracted": 28950,
        "average_processing_time": 8.3
    },
    "processing_metrics": {
        "total_runs": 12,
        "last_processing_time": "2024-01-14T10:30:00Z",
        "total_communities": 45,
        "average_processing_time": 45.2
    },
    "system_health": {
        "database_connected": true,
        "llm_api_available": true,
        "disk_usage_percent": 65.2,
        "memory_usage_percent": 42.1
    }
}
```

---

## HealthChecker

System health monitoring and validation (currently placeholder implementation).

### Class Signature

```python
class HealthChecker:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def check_database_connection(self) -> bool
    async def check_llm_api(self) -> bool
    async def check_disk_space(self) -> Dict[str, Any]
    async def check_memory_usage(self) -> Dict[str, Any]
    async def run_full_health_check(self) -> Dict[str, Any]
```

### Methods

#### check_database_connection()

Verify PostgreSQL database connectivity.

```python
async def check_database_connection(self) -> bool:
    """Check database connection health."""
```

**Returns**: True if database is accessible

#### check_llm_api()

Verify LLM API availability.

```python
async def check_llm_api(self) -> bool:
    """Check LLM API health."""
```

**Returns**: True if LLM API responds

#### check_disk_space()

Monitor disk space usage.

```python
async def check_disk_space(self) -> Dict[str, Any]:
    """Check disk space usage."""
```

**Returns**: Disk usage statistics

```python
{
    "total_gb": 500.0,
    "used_gb": 325.0,
    "free_gb": 175.0,
    "usage_percent": 65.0,
    "status": "healthy"  # or "warning", "critical"
}
```

#### check_memory_usage()

Monitor memory usage.

```python
async def check_memory_usage(self) -> Dict[str, Any]:
    """Check memory usage."""
```

**Returns**: Memory usage statistics

```python
{
    "total_gb": 16.0,
    "used_gb": 6.7,
    "free_gb": 9.3,
    "usage_percent": 41.9,
    "status": "healthy"
}
```

#### run_full_health_check()

Execute comprehensive health assessment.

```python
async def run_full_health_check(self) -> Dict[str, Any]:
    """Run complete health check."""
```

**Returns**: Full health status

```python
{
    "timestamp": "2024-01-14T10:30:00Z",
    "overall_status": "healthy",  # healthy, degraded, critical
    "checks": {
        "database": {
            "status": "healthy",
            "response_time_ms": 5.2,
            "details": "Connection successful"
        },
        "llm_api": {
            "status": "healthy",
            "response_time_ms": 234.1,
            "details": "API responding"
        },
        "disk_space": {
            "status": "healthy",
            "usage_percent": 65.2,
            "details": "Sufficient space available"
        },
        "memory": {
            "status": "healthy",
            "usage_percent": 42.1,
            "details": "Memory usage normal"
        }
    },
    "recommendations": []
}
```

---

## Logging Setup

Structured logging configuration for the research agent.

### setup_logging()

Configure logging for all agent components.

```python
def setup_logging(
    config: Config,
    debug: bool = False
) -> Tuple[logging.Logger, logging.Logger, logging.Logger, logging.Logger]:
    """Setup structured logging for all components."""
```

**Parameters**:
- `config`: Configuration object with logging settings
- `debug`: Enable debug-level logging

**Returns**: Tuple of configured loggers:
- `agent_logger`: Main agent operations
- `ingestion_logger`: Content ingestion operations
- `processing_logger`: Graph processing operations
- `orchestrator_logger`: Task scheduling operations

### Log Structure

All logs follow structured format:

```json
{
    "timestamp": "2024-01-14T10:30:15.123Z",
    "level": "INFO",
    "component": "ingestion",
    "operation": "extract_entities",
    "task_id": 1234,
    "duration_ms": 1250,
    "entities_count": 15,
    "message": "Successfully extracted entities from content"
}
```

### Log Files

Separate log files for each component:

- `logs/agent.log`: Main agent operations and errors
- `logs/ingestion.log`: Content ingestion pipeline
- `logs/processing.log`: Community detection and analysis
- `logs/orchestrator.log`: Task scheduling and queue management

### Log Rotation

Automatic log rotation based on configuration:

```yaml
monitoring:
  log_level: "INFO"
  log_file_max_size_mb: 10
  log_file_backup_count: 5
```

---

## Usage Examples

### Basic Metrics Collection

```python
from research_agent.monitoring import MetricsCollector

metrics = MetricsCollector(config)

# Record task lifecycle
await metrics.record_task_start(task_id=123, task_type="INGEST")
# ... task execution ...
await metrics.record_task_complete(
    task_id=123,
    duration_seconds=8.5,
    success=True
)

# Record ingestion metrics
await metrics.record_ingestion_start(content_length=50000)
await metrics.record_ingestion_complete(
    entities_count=25,
    relationships_count=45,
    duration_seconds=8.5
)
```

### Health Monitoring

```python
from research_agent.monitoring import HealthChecker

health = HealthChecker(config)

# Quick health checks
db_ok = await health.check_database_connection()
llm_ok = await health.check_llm_api()

print(f"Database: {'✓' if db_ok else '✗'}")
print(f"LLM API: {'✓' if llm_ok else '✗'}")

# Full health assessment
health_status = await health.run_full_health_check()
print(f"Overall health: {health_status['overall_status']}")

for check_name, check_result in health_status['checks'].items():
    status_icon = "✓" if check_result['status'] == 'healthy' else "✗"
    print(f"{check_name}: {status_icon} ({check_result['response_time_ms']}ms)")
```

### Logging Setup

```python
from research_agent.monitoring import setup_logging

# Setup logging for all components
agent_logger, ingestion_logger, processing_logger, orchestrator_logger = setup_logging(
    config,
    debug=True  # Enable debug logging
)

# Use loggers in code
agent_logger.info("Agent started successfully")
ingestion_logger.debug(f"Processing content of length {len(content)}")
processing_logger.warning("Community detection taking longer than expected")
orchestrator_logger.error(f"Task {task_id} failed: {error_message}")
```

### Metrics Analysis

```python
# Get comprehensive metrics
summary = await metrics.get_summary_metrics()

# Task performance
task_metrics = summary['task_metrics']
print(f"Task success rate: {task_metrics['success_rate']}%")
print(f"Average task duration: {task_metrics['average_duration']}s")

# Ingestion statistics
ingestion_metrics = summary['ingestion_metrics']
print(f"Total entities extracted: {ingestion_metrics['entities_extracted']}")
print(f"Average processing time: {ingestion_metrics['average_processing_time']}s")

# System health
health = summary['system_health']
print(f"Database connected: {health['database_connected']}")
print(f"Disk usage: {health['disk_usage_percent']}%")
```

### Log Analysis

```bash
# View recent agent logs
tail -f logs/agent.log

# Search for errors
grep "ERROR" logs/*.log

# Analyze task performance
grep "task_id" logs/orchestrator.log | head -10

# Monitor ingestion throughput
grep "ingestion_complete" logs/ingestion.log | tail -5
```

---

## Configuration

### Monitoring Configuration

```yaml
monitoring:
  log_level: "INFO"
  log_file_max_size_mb: 10
  log_file_backup_count: 5

  health_checks:
    enabled: true
    interval_minutes: 5

  metrics:
    enabled: true
    collection_interval_seconds: 60
```

### Logging Configuration

```yaml
monitoring:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file_max_size_mb: 10
  log_file_backup_count: 5

paths:
  logs: "./logs"
```

### Health Check Thresholds

```yaml
monitoring:
  health_thresholds:
    max_disk_usage_percent: 90
    max_memory_usage_percent: 85
    max_response_time_ms: 5000
    min_success_rate_percent: 95
```

---

## Monitoring Dashboard

### Real-time Metrics

The system provides real-time monitoring through:

- **Task Queue Status**: Pending, running, completed, failed tasks
- **Performance Metrics**: Throughput, latency, success rates
- **Resource Usage**: CPU, memory, disk, network
- **Health Status**: Component availability and response times

### Alerting

Configure alerts for critical conditions:

- Task failure rate > 10%
- Database connection lost
- LLM API unavailable
- Disk space < 10% free
- Memory usage > 90%

### Historical Analysis

Track trends over time:

- Daily task completion rates
- Weekly ingestion volumes
- Monthly community growth
- Quarterly performance improvements

---

## Troubleshooting

### Logging Issues

**Problem**: Logs not appearing

**Solutions**:
```python
# Check logging setup
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Verify log directory exists
import os
os.makedirs("logs", exist_ok=True)

# Check file permissions
os.access("logs", os.W_OK)
```

### Metrics Issues

**Problem**: Metrics not collecting

**Solutions**:
```python
# Verify metrics collector initialization
metrics = MetricsCollector(config)
print(f"Metrics enabled: {config.monitoring.metrics.enabled}")

# Check database connectivity for metrics storage
# Metrics are stored in agent_state table
```

### Health Check Issues

**Problem**: Health checks failing

**Solutions**:
```python
# Test individual checks
db_ok = await health.check_database_connection()
print(f"Database check: {db_ok}")

llm_ok = await health.check_llm_api()
print(f"LLM check: {llm_ok}")

# Check configuration
print(f"Health checks enabled: {config.monitoring.health_checks.enabled}")
```

### Performance Monitoring

**Problem**: High resource usage

**Solutions**:
```python
# Monitor resource usage
import psutil

cpu = psutil.cpu_percent()
memory = psutil.virtual_memory()
disk = psutil.disk_usage('/')

print(f"CPU: {cpu}%")
print(f"Memory: {memory.percent}%")
print(f"Disk: {disk.percent}%")
```

---

## Performance Tuning

### Logging Optimization

- **Log Levels**: Use appropriate levels (INFO for production)
- **Structured Logging**: JSON format for better parsing
- **Log Rotation**: Prevent disk space issues
- **Async Logging**: Non-blocking log writes

### Metrics Optimization

- **Batch Updates**: Collect metrics in batches
- **Sampling**: Sample high-frequency metrics
- **Retention**: Configure metric retention periods
- **Compression**: Compress historical metrics

### Health Check Optimization

- **Check Frequency**: Balance monitoring vs. overhead
- **Timeout Settings**: Appropriate timeouts for checks
- **Parallel Checks**: Run health checks concurrently
- **Caching**: Cache results for short periods

---

## Integration with External Monitoring

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'research-agent'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### Grafana Dashboard

Create dashboards for:

- Task queue monitoring
- Ingestion pipeline performance
- Community detection metrics
- System resource usage
- LLM API usage statistics

### Alert Manager

Configure alerts for:

- Task failure spikes
- Performance degradation
- Resource exhaustion
- API unavailability

---

**Last Updated**: January 14, 2026