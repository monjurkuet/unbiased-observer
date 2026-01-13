# Configuration Guide

Complete configuration reference for the Autonomous Research Agent system.

---

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [YAML Configuration Files](#yaml-configuration-files)
  - [research_agent_config.yaml](#research_agent_configyaml)
  - [research_sources.yaml](#research_sourcesyaml)
- [Configuration Sections](#configuration-sections)
  - [Database Configuration](#database-configuration)
  - [LLM Configuration](#llm-configuration)
  - [Research Configuration](#research-configuration)
  - [Ingestion Configuration](#ingestion-configuration)
  - [Processing Configuration](#processing-configuration)
  - [Monitoring Configuration](#monitoring-configuration)
  - [Paths Configuration](#paths-configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Autonomous Research Agent uses a two-tier configuration system:

1. **Environment Variables** - For sensitive data and runtime settings
2. **YAML Configuration Files** - For structured configuration of agent behavior

```
Configuration Priority (highest to lowest):
1. Environment variables
2. research_agent_config.yaml
3. Default values in code
```

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google GenAI API key for LLM operations | `AIzaSy...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DEBUG` | Enable debug logging (1=enabled, 0=disabled) | `0` | `1` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` | `DEBUG` |
| `DB_CONNECTION_STRING` | PostgreSQL connection string | From config | `postgresql://user:pass@localhost:5432/db` |
| `LLM_BASE_URL` | LLM API base URL | From config | `http://localhost:8317/v1` |
| `LLM_API_KEY` | LLM API authentication key | From config | `lm-studio` |
| `CONFIG_PATH` | Custom path to configuration file | `configs/research_agent_config.yaml` | `/custom/path/config.yaml` |
| `SOURCES_PATH` | Custom path to sources configuration | `configs/research_sources.yaml` | `/custom/path/sources.yaml` |

### Setting Environment Variables

#### Method 1: .env File (Recommended)

Create a `.env` file in the project root:

```bash
# .env
GOOGLE_API_KEY=your_google_genai_key_here
DEBUG=0
LOG_LEVEL=INFO
DB_CONNECTION_STRING=postgresql://agentzero:password@localhost:5432/knowledge_graph
LLM_BASE_URL=http://localhost:8317/v1
LLM_API_KEY=lm-studio
```

Load the environment variables:

```bash
source .env
```

#### Method 2: Shell Export

```bash
export GOOGLE_API_KEY="your_google_genai_key_here"
export DEBUG=1
export LOG_LEVEL=DEBUG
```

#### Method 3: Systemd Service

For production deployment (see [Deployment Guide](deployment.md)):

```ini
# /etc/systemd/system/research-agent.service
[Service]
Environment="GOOGLE_API_KEY=your_google_genai_key_here"
Environment="DEBUG=0"
Environment="LOG_LEVEL=INFO"
```

---

## YAML Configuration Files

### research_agent_config.yaml

Main configuration file for the Autonomous Research Agent.

**Location**: `configs/research_agent_config.yaml`

**Example Configuration**:

```yaml
# Database Configuration
database:
  connection_string: "postgresql://agentzero@localhost:5432/knowledge_graph"
  pool_min_size: 5
  pool_max_size: 20

# LLM Configuration
llm:
  base_url: "http://localhost:8317/v1"
  api_key: "lm-studio"
  model_default: "gemini-2.5-flash"
  model_pro: "gemini-2.5-pro"
  max_retries: 3
  timeout: 120  # seconds

# Embedding Configuration
embedding:
  model: "text-embedding-004"
  dimension: 768
  batch_size: 100

# Research Configuration
research:
  arxiv:
    monitoring_enabled: true
    monitoring_interval_hours: 2
    search_configs:
      - name: "AI Research"
        keywords: ["artificial intelligence", "machine learning"]
        max_results: 5
        days_back: 7
        active: true
      - name: "Computer Vision"
        keywords: ["computer vision", "image recognition"]
        max_results: 3
        days_back: 3
        active: true

  rate_limit:
    requests_per_minute: 30
    concurrent_requests: 5
    backoff_seconds: 2

# Ingestion Configuration
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

# Processing Configuration
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

# Monitoring Configuration
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

# Paths Configuration
paths:
  logs: "./logs"
  cache: "./cache"
  temp: "./temp"
  data: "./data"

  # Ensure directories exist on startup
  auto_create: true
```

### research_sources.yaml

Research source definitions for automated content discovery.

**Location**: `configs/research_sources.yaml`

**Example Configuration**:

```yaml
sources:
  # arXiv Automated Sources (handled by ArxivSourceManager)
  - type: "arxiv"
    name: "ArXiv AI Research"
    config:
      category: "cs.AI"
      max_results: 5
      days_back: 7
    active: true
    priority: "high"

  - type: "arxiv"
    name: "ArXiv Machine Learning"
    config:
      keywords: ["machine learning", "deep learning"]
      max_results: 10
      days_back: 3
    active: true
    priority: "high"

  # RSS Feeds
  - type: "rss"
    name: "Nature Machine Intelligence"
    config:
      url: "https://www.nature.com/nmachintelligence.rss"
      fetch_interval_hours: 24
      max_articles: 5
    active: false
    priority: "medium"

  - type: "rss"
    name: "OpenAI Blog"
    config:
      url: "https://openai.com/blog/rss.xml"
      fetch_interval_hours: 48
      max_articles: 3
    active: false
    priority: "medium"

  # Custom URLs
  - type: "url"
    name: "Important Research Paper"
    config:
      url: "https://arxiv.org/pdf/2301.07041.pdf"
      metadata:
        category: "transformers"
        priority: "high"
    active: true
    priority: "high"

  # File Sources
  - type: "file"
    name: "Local Research Paper"
    config:
      path: "./data/papers/my_research.pdf"
      metadata:
        category: "custom"
        author: "Researcher Name"
    active: false
    priority: "low"

# Source Priority Levels
# - high: Process immediately, fetch every 1-2 hours
# - medium: Process regularly, fetch every 6-12 hours
# - low: Process occasionally, fetch every 24 hours
```

---

## Configuration Sections

### Database Configuration

Controls PostgreSQL connection pooling and database behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_string` | string | Required | PostgreSQL connection URL |
| `pool_min_size` | int | 5 | Minimum connection pool size |
| `pool_max_size` | int | 20 | Maximum connection pool size |

**Connection String Format**:

```bash
postgresql://[user[:password]@][host][:port][/database][?parameters]

# Examples:
postgresql://agentzero@localhost:5432/knowledge_graph
postgresql://agentzero:password@db.example.com:5432/knowledge_graph?sslmode=require
```

### LLM Configuration

Controls LLM API connection and model selection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | Required | LLM API base URL |
| `api_key` | string | Required | LLM API authentication key |
| `model_default` | string | `gemini-2.5-flash` | Default model for operations |
| `model_pro` | string | `gemini-2.5-pro` | High-quality model for summarization |
| `max_retries` | int | 3 | Maximum retry attempts for API calls |
| `timeout` | int | 120 | Request timeout in seconds |

**Supported Models**:

- `gemini-2.5-flash`: Fast, cost-effective for entity extraction
- `gemini-2.5-pro`: High-quality for summarization
- `text-embedding-004`: Embedding generation for semantic search

### Embedding Configuration

Controls vector embedding generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `text-embedding-004` | Embedding model name |
| `dimension` | int | 768 | Embedding vector dimension |
| `batch_size` | int | 100 | Number of embeddings per API call |

### Research Configuration

Controls research source discovery and monitoring.

#### arxiv Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitoring_enabled` | bool | `true` | Enable arXiv monitoring |
| `monitoring_interval_hours` | int | 2 | Hours between monitoring cycles |
| `search_configs` | array | - | List of search configurations |

#### search_configs

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Configuration name |
| `keywords` | array | Keywords to search for |
| `max_results` | int | Maximum papers to fetch |
| `days_back` | int | Number of days back to search |
| `active` | bool | Enable/disable this config |

#### rate_limit Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_per_minute` | int | 30 | Maximum API requests per minute |
| `concurrent_requests` | int | 5 | Maximum concurrent requests |
| `backoff_seconds` | int | 2 | Backoff time between requests |

### Ingestion Configuration

Controls content ingestion and entity extraction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_content_length` | int | 100000 | Maximum content characters |
| `max_retries` | int | 3 | Maximum retry attempts |
| `batch_size` | int | 10 | Batch size for storage |
| `concurrent_ingestions` | int | 3 | Max concurrent ingestion tasks |

#### content_types Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Content type (pdf, html, markdown, text) |
| `max_size_mb` | int | Maximum file size in MB |
| `enabled` | bool | Enable/disable content type |

### Processing Configuration

Controls community detection and summarization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable processing pipeline |
| `min_entities_to_process` | int | 100 | Minimum entities before processing |
| `min_time_between_processing_hours` | int | 6 | Minimum time between processing |

#### community_detection Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | `leiden` | Community detection algorithm |
| `resolution` | float | 1.0 | Resolution parameter for community size |
| `max_levels` | int | 10 | Maximum hierarchy levels |

#### summarization Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable community summarization |
| `max_communities_per_batch` | int | 10 | Max communities per batch |

### Monitoring Configuration

Controls logging, health checks, and metrics.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | string | `INFO` | Logging level |
| `log_file_max_size_mb` | int | 10 | Maximum log file size |
| `log_file_backup_count` | int | 5 | Number of log backups |

#### health_checks Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable health checks |
| `interval_minutes` | int | 5 | Health check interval |

#### metrics Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable metrics collection |
| `collection_interval_seconds` | int | 60 | Metrics collection interval |

### Paths Configuration

Controls file system paths and directory management.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logs` | string | `./logs` | Log files directory |
| `cache` | string | `./cache` | Cache directory |
| `temp` | string | `./temp` | Temporary files directory |
| `data` | string | `./data` | Data files directory |
| `auto_create` | bool | `true` | Auto-create directories on startup |

---

## Best Practices

### Security

1. **Never commit sensitive data**
   - Use `.env` for API keys
   - Add `.env` to `.gitignore`
   - Use environment variables in production

2. **Use strong database passwords**
   ```yaml
   # Good: Use environment variable
   database:
     connection_string: "${DB_CONNECTION_STRING}"

   # Bad: Hardcoded password
   database:
     connection_string: "postgresql://user:weakpassword@..."
   ```

3. **Restrict database access**
   - Use database-specific users with limited permissions
   - Enable SSL for remote connections
   - Regularly rotate passwords

### Performance

1. **Tune connection pooling**
   ```yaml
   database:
     pool_min_size: 5   # Start with 5 connections
     pool_max_size: 20  # Scale to 20 under load
   ```

2. **Optimize ingestion batch size**
   ```yaml
   ingestion:
     batch_size: 10  # Balance between memory and I/O
     concurrent_ingestions: 3  # Don't overwhelm LLM API
   ```

3. **Adjust rate limiting**
   ```yaml
   research:
     rate_limit:
       requests_per_minute: 30  # Respect API limits
       concurrent_requests: 5   # Prevent timeouts
   ```

### Reliability

1. **Configure appropriate retry limits**
   ```yaml
   llm:
     max_retries: 3  # Balance between persistence and latency

   ingestion:
     max_retries: 3  # Retry transient failures
   ```

2. **Enable monitoring in production**
   ```yaml
   monitoring:
     health_checks:
       enabled: true  # Detect issues early
       interval_minutes: 5
   ```

3. **Set reasonable processing thresholds**
   ```yaml
   processing:
     min_entities_to_process: 100  # Only process with sufficient data
     min_time_between_processing_hours: 6  # Avoid over-processing
   ```

### Development vs Production

**Development Configuration**:
```yaml
llm:
  model_default: "gemini-2.5-flash"  # Fast for testing

monitoring:
  log_level: "DEBUG"  # Verbose logging

processing:
  min_entities_to_process: 10  # Low threshold for testing
```

**Production Configuration**:
```yaml
llm:
  model_default: "gemini-2.5-pro"  # High-quality for production

monitoring:
  log_level: "INFO"  # Efficient logging

processing:
  min_entities_to_process: 100  # High threshold for efficiency
```

---

## Troubleshooting

### Configuration Not Loading

**Problem**: Agent uses default values instead of configured values

**Solution**:
```bash
# Check configuration file path
export CONFIG_PATH="/absolute/path/to/config.yaml"

# Verify YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/research_agent_config.yaml'))"

# Check for syntax errors (indentation, quotes)
cat configs/research_agent_config.yaml
```

### Environment Variables Not Working

**Problem**: Environment variables ignored in production

**Solution**:
```bash
# Verify variables are set
echo $GOOGLE_API_KEY
echo $DB_CONNECTION_STRING

# Source .env file
source .env

# Check systemd service environment
sudo systemctl show research-agent | grep Environment
```

### Database Connection Issues

**Problem**: FATAL: password authentication failed

**Solution**:
```yaml
# Use environment variable for connection string
database:
  connection_string: "${DB_CONNECTION_STRING}"

# Set in .env
echo 'DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/db' >> .env
```

### LLM API Connection Issues

**Problem**: Connection timeout or API errors

**Solution**:
```yaml
llm:
  base_url: "http://localhost:8317/v1"  # Verify URL is correct
  timeout: 120  # Increase timeout if needed
  max_retries: 3  # Ensure retries are enabled

# Set environment variable
export LLM_BASE_URL="http://localhost:8317/v1"
```

### Processing Not Running

**Problem**: Community detection not triggering

**Solution**:
```yaml
processing:
  enabled: true  # Must be enabled
  min_entities_to_process: 100  # Check if threshold is too high

# Temporarily lower for testing
min_entities_to_process: 10
min_time_between_processing_hours: 1
```

---

## Next Steps

After configuration:

1. **Test configuration**: See [Installation Guide - Verification](installation.md#step-7-verify-installation)
2. **Start agent**: `PYTHONPATH=/home/administrator/dev/unbiased-observer python3 research_agent/main.py`
3. **Monitor logs**: `tail -f ./logs/agent.log`
4. **Deploy to production**: See [Deployment Guide](deployment.md)

---

**Last Updated**: January 14, 2026
