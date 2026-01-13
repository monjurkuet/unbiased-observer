# Troubleshooting Guide

Common issues and solutions for the Autonomous Research Agent.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Runtime Errors](#runtime-errors)
- [Database Issues](#database-issues)
- [API and Network Problems](#api-and-network-problems)
- [Performance Issues](#performance-issues)
- [Processing Problems](#processing-problems)
- [Web Interface Issues](#web-interface-issues)
- [Monitoring and Logging](#monitoring-and-logging)
- [Recovery Procedures](#recovery-procedures)

---

## Installation Issues

### Python Version Problems

**Problem**: `python3 --version` shows wrong version

**Symptoms**:
- Import errors for modern Python features
- Installation fails with syntax errors

**Solutions**:

```bash
# Check available Python versions
ls /usr/bin/python*

# Install Python 3.10 if missing
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-venv python3.10-dev

# Create virtual environment with correct Python
python3.10 -m venv venv
source venv/bin/activate
python --version  # Should show Python 3.10.x
```

### Dependency Installation Failures

**Problem**: `pip install` fails with compilation errors

**Symptoms**:
- GCC compilation errors
- Missing header files
- Permission denied errors

**Solutions**:

```bash
# Install build dependencies
sudo apt install build-essential python3-dev

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with verbose output
pip install -r requirements.txt -v

# If still failing, try without cache
pip install --no-cache-dir -r requirements.txt
```

### PostgreSQL Installation Issues

**Problem**: PostgreSQL service not starting

**Symptoms**:
- `sudo systemctl status postgresql` shows failed
- Connection refused errors

**Solutions**:

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# View error logs
sudo journalctl -u postgresql -n 50

# Try starting manually
sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main start

# Reinitialize if corrupted
sudo pg_dropcluster 14 main
sudo pg_createcluster 14 main
sudo systemctl start postgresql
```

### pgvector Extension Issues

**Problem**: `CREATE EXTENSION vector` fails

**Symptoms**:
- Extension not found error
- Vector operations fail

**Solutions**:

```bash
# Check if pgvector is installed
dpkg -l | grep pgvector

# Install pgvector for PostgreSQL 14
sudo apt install postgresql-14-pgvector

# Or compile from source
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Restart PostgreSQL
sudo systemctl restart postgresql

# Verify extension
psql -U postgres -d knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## Configuration Problems

### Environment Variables Not Set

**Problem**: Agent ignores configuration settings

**Symptoms**:
- Default values used instead of configured values
- API keys not found errors

**Solutions**:

```bash
# Check if .env file exists
ls -la .env

# Verify environment variables
cat .env

# Source environment file
source .env

# Check variables are set
echo $GOOGLE_API_KEY
echo $DB_CONNECTION_STRING

# For systemd services, check service file
sudo systemctl cat research-agent | grep Environment
```

### YAML Configuration Errors

**Problem**: Configuration file parsing fails

**Symptoms**:
- Agent fails to start with YAML errors
- Settings not applied

**Solutions**:

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/research_agent_config.yaml'))"

# Check indentation (YAML is sensitive to spaces)
cat -n configs/research_agent_config.yaml

# Common issues:
# - Wrong indentation level
# - Missing quotes around strings with special characters
# - Wrong boolean values (True/False vs true/false)

# Test configuration loading
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
from research_agent.config import load_config
try:
    config = load_config()
    print('Configuration loaded successfully')
    print(f'Database: {config.database.connection_string[:50]}...')
except Exception as e:
    print(f'Configuration error: {e}')
"
```

### Database Connection String Issues

**Problem**: Database connection fails

**Symptoms**:
- Connection refused
- Authentication failed
- Database does not exist

**Solutions**:

```bash
# Test basic connection
psql -U postgres -h localhost -c "SELECT version();"

# Check database exists
psql -U postgres -l | grep knowledge_graph

# Verify user permissions
psql -U postgres -c "SELECT usename, usecreatedb, usesuper FROM pg_user WHERE usename = 'research_agent';"

# Test with application user
psql -U research_agent -d knowledge_graph -c "SELECT 1;"

# Check connection string format
# Correct: postgresql://user:password@host:port/database
# Wrong: postgresql://user:password@host/database (missing port)
```

---

## Runtime Errors

### Import Errors

**Problem**: Module not found errors on startup

**Symptoms**:
- `ImportError: No module named 'research_agent'`
- Agent fails to start

**Solutions**:

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Set correct PYTHONPATH
export PYTHONPATH=/home/administrator/dev/unbiased-observer

# Add to .bashrc for persistence
echo 'export PYTHONPATH=/home/administrator/dev/unbiased-observer' >> ~/.bashrc

# Test imports
python3 -c "import research_agent.config; print('Import successful')"

# Check virtual environment
which python3
# Should point to venv/bin/python3

# Reinstall if corrupted
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### AsyncIO Errors

**Problem**: RuntimeWarning about asyncio

**Symptoms**:
- Warnings about event loop
- Tasks not executing properly

**Solutions**:

```python
# Ensure proper asyncio usage
import asyncio

# Correct pattern
async def main():
    # Your async code here
    pass

if __name__ == "__main__":
    asyncio.run(main())

# Avoid mixing sync and async code
# Don't use time.sleep() in async functions
```

### Memory Errors

**Problem**: Out of memory during processing

**Symptoms**:
- Process killed by OOM killer
- MemoryError exceptions

**Solutions**:

```yaml
# Reduce batch sizes in configuration
ingestion:
  batch_size: 5  # Reduce from 10

processing:
  min_entities_to_process: 50  # Reduce from 100

# Monitor memory usage
import psutil
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")

# Add memory limits to systemd
sudo nano /etc/systemd/system/research-agent.service
# Add: MemoryLimit=4G
sudo systemctl daemon-reload
sudo systemctl restart research-agent
```

---

## Database Issues

### Connection Pool Exhaustion

**Problem**: Too many database connections

**Symptoms**:
- Connection pool exhausted errors
- Slow database queries

**Solutions**:

```yaml
# Adjust connection pool settings
database:
  pool_min_size: 2  # Reduce minimum connections
  pool_max_size: 10  # Reduce maximum connections

# Monitor active connections
psql -U postgres -d knowledge_graph -c "
SELECT count(*) as active_connections
FROM pg_stat_activity
WHERE datname = 'knowledge_graph';
"

# Check for connection leaks
psql -U postgres -d knowledge_graph -c "
SELECT state, count(*)
FROM pg_stat_activity
WHERE datname = 'knowledge_graph'
GROUP BY state;
"
```

### Table/Index Issues

**Problem**: Database queries are slow

**Symptoms**:
- Long query execution times
- High CPU usage on database

**Solutions**:

```sql
-- Check existing indexes
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename;

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM nodes WHERE type = 'concept' LIMIT 10;

-- Add missing indexes if needed
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_edges_source_target ON edges(source_id, target_id);

-- Update statistics
VACUUM ANALYZE;
```

### Data Corruption

**Problem**: Inconsistent or corrupted data

**Symptoms**:
- Foreign key constraint violations
- Missing relationships
- Orphaned records

**Solutions**:

```sql
-- Check for orphaned edges
SELECT COUNT(*) as orphaned_edges
FROM edges e
LEFT JOIN nodes n1 ON e.source_id = n1.id
LEFT JOIN nodes n2 ON e.target_id = n2.id
WHERE n1.id IS NULL OR n2.id IS NULL;

-- Check for duplicate entities
SELECT name, type, COUNT(*) as count
FROM nodes
GROUP BY name, type
HAVING COUNT(*) > 1;

-- Repair data (CAUTION - backup first)
-- Remove orphaned edges
DELETE FROM edges
WHERE source_id NOT IN (SELECT id FROM nodes)
   OR target_id NOT IN (SELECT id FROM nodes);
```

---

## API and Network Problems

### Google GenAI API Issues

**Problem**: LLM API calls failing

**Symptoms**:
- API quota exceeded
- Invalid API key
- Rate limit errors

**Solutions**:

```bash
# Test API key
curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models?key=$GOOGLE_API_KEY"

# Check API quota
# Visit: https://makersuite.google.com/app/apikey

# Adjust rate limits in configuration
llm:
  max_retries: 5
  timeout: 60  # Increase timeout

# Switch to different model
llm:
  model_default: "gemini-2.5-flash"  # Faster, cheaper
  model_pro: "gemini-2.5-pro"        # Slower, better quality
```

### arXiv API Issues

**Problem**: arXiv discovery not working

**Symptoms**:
- No new papers found
- API timeout errors

**Solutions**:

```bash
# Test arXiv API manually
curl "http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=1"

# Check network connectivity
ping export.arxiv.org

# Adjust search parameters
research:
  arxiv:
    search_configs:
      - name: "AI Research"
        keywords: ["artificial intelligence"]  # Simpler keywords
        max_results: 3  # Reduce batch size
        days_back: 3    # Shorter time window
```

### Content Fetching Issues

**Problem**: URL fetching fails

**Symptoms**:
- HTTP errors
- SSL certificate issues
- Rate limiting

**Solutions**:

```bash
# Test URL accessibility
curl -I "https://example.com/paper.pdf"

# Check SSL certificates
curl -v "https://example.com/paper.pdf" 2>&1 | grep -A 5 "SSL certificate"

# Adjust rate limits
research:
  rate_limit:
    requests_per_minute: 20  # Reduce from 30
    concurrent_requests: 3   # Reduce from 5

# Add retry logic
research:
  max_retries: 3
  backoff_seconds: 2
```

---

## Performance Issues

### High CPU Usage

**Problem**: Agent consuming too much CPU

**Symptoms**:
- System slowdown
- High CPU percentages

**Solutions**:

```bash
# Monitor CPU usage
top -p $(pgrep -f research_agent)

# Adjust processing intervals
orchestrator:
  queue_processing_interval: 20  # Increase from 10 seconds

# Reduce concurrent operations
ingestion:
  concurrent_ingestions: 2  # Reduce from 3

research:
  rate_limit:
    concurrent_requests: 3  # Reduce from 5
```

### Slow Processing

**Problem**: Tasks taking too long to complete

**Symptoms**:
- Long processing times
- Queue backlog growing

**Solutions**:

```yaml
# Optimize batch sizes
ingestion:
  batch_size: 5  # Smaller batches

# Adjust processing thresholds
processing:
  min_entities_to_process: 50  # Process more frequently

# Monitor slow operations
# Check logs for timing information
grep "processing_time_seconds" logs/ingestion.log | tail -10
```

### Memory Leaks

**Problem**: Memory usage growing over time

**Symptoms**:
- Increasing memory consumption
- Out of memory errors

**Solutions**:

```python
# Monitor memory usage
import psutil
import gc

def check_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

    # Force garbage collection
    gc.collect()

# Add memory monitoring to your code
```

### Database Performance

**Problem**: Slow database queries

**Symptoms**:
- Long query times
- Database CPU high

**Solutions**:

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT COUNT(*) FROM nodes;

-- Check table statistics
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
FROM pg_stat_user_tables
ORDER BY n_tup_ins DESC;

-- Adjust PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '64MB';
SELECT pg_reload_conf();
```

---

## Processing Problems

### Community Detection Issues

**Problem**: Community detection not running

**Symptoms**:
- No communities created
- Processing logs show errors

**Solutions**:

```yaml
# Check processing configuration
processing:
  enabled: true
  min_entities_to_process: 50  # Ensure threshold is met

# Verify entity count
psql -U research_agent -d knowledge_graph -c "SELECT COUNT(*) FROM nodes;"

# Check processing logs
tail -20 logs/processing.log

# Test community detection manually
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
from knowledge_base.community import CommunityDetector
import asyncio

async def test():
    detector = CommunityDetector()
    graph = await detector.load_graph()
    communities = await detector.detect_communities()
    print(f'Found {len(communities)} communities')

asyncio.run(test())
"
```

### LLM Processing Errors

**Problem**: Entity extraction failing

**Symptoms**:
- Empty entity lists
- LLM API errors

**Solutions**:

```yaml
# Check LLM configuration
llm:
  model_default: "gemini-2.5-flash"  # Ensure valid model
  max_retries: 3

# Test LLM API
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
import google.generativeai as genai
genai.configure(api_key='$GOOGLE_API_KEY')
model = genai.GenerativeModel('gemini-2.5-flash')
response = model.generate_content('Test message')
print('LLM API working')
"

# Adjust content limits
ingestion:
  max_content_length: 50000  # Reduce if needed
```

---

## Web Interface Issues

### Streamlit Not Starting

**Problem**: UI fails to start

**Symptoms**:
- Port 8501 not accessible
- Streamlit errors

**Solutions**:

```bash
# Check Streamlit installation
streamlit --version

# Test basic Streamlit app
cd ui
python3 -c "import streamlit as st; print('Streamlit OK')"

# Start with debug
streamlit run app.py --logger.level debug

# Check port availability
netstat -tlnp | grep 8501
```

### Data Loading Issues

**Problem**: UI shows no data or mock data

**Symptoms**:
- Empty graphs
- Mock statistics displayed

**Solutions**:

```python
# Check database connection in UI
cd ui
python3 -c "
from research_agent.config import load_config
config = load_config()
print('Config loaded')
# Test database connection
"

# Verify data exists
psql -U research_agent -d knowledge_graph -c "
SELECT COUNT(*) as nodes FROM nodes;
SELECT COUNT(*) as edges FROM edges;
SELECT COUNT(*) as communities FROM communities;
"
```

### Visualization Issues

**Problem**: Network graph not rendering

**Symptoms**:
- No graph displayed
- Plotly errors

**Solutions**:

```python
# Check Plotly installation
import plotly.graph_objects as go
print("Plotly available")

# Test graph creation
fig = create_network_graph(get_mock_graph_data())
print(f"Graph created with {len(fig.data)} traces")
```

### Performance Issues

**Problem**: UI slow or unresponsive

**Symptoms**:
- Long loading times
- Browser timeouts

**Solutions**:

```python
# Enable caching
@st.cache_data(ttl=300)
def load_cached_stats():
    return get_mock_stats()

# Reduce data size
graph_data = get_mock_graph_data()
if len(graph_data['nodes']) > 500:
    graph_data['nodes'] = graph_data['nodes'][:500]  # Limit nodes
```

---

## Monitoring and Logging

### Log Files Not Created

**Problem**: No log files in logs/ directory

**Symptoms**:
- Missing log files
- No logging output

**Solutions**:

```bash
# Check directory permissions
ls -la logs/

# Create logs directory
mkdir -p logs
chmod 755 logs

# Check logging configuration
python3 -c "
from research_agent.monitoring import setup_logging
from research_agent.config import load_config
config = load_config()
loggers = setup_logging(config)
print('Logging configured')
"
```

### Log Rotation Issues

**Problem**: Log files not rotating

**Symptoms**:
- Huge log files
- Disk space issues

**Solutions**:

```bash
# Check logrotate configuration
cat /etc/logrotate.d/research-agent

# Test log rotation
sudo logrotate -f /etc/logrotate.d/research-agent

# Adjust rotation settings
sudo nano /etc/logrotate.d/research-agent
# Modify size/compression settings
sudo systemctl restart logrotate
```

### Metrics Not Collecting

**Problem**: No metrics data

**Symptoms**:
- Empty metrics output
- Missing performance data

**Solutions**:

```python
# Test metrics collection
from research_agent.monitoring import MetricsCollector
import asyncio

async def test_metrics():
    metrics = MetricsCollector()
    await metrics.record_task_start(1, 'test')
    await metrics.record_task_complete(1, 5.0, True)
    
    summary = await metrics.get_summary_metrics()
    print(f"Metrics: {summary}")

asyncio.run(test_metrics())
```

---

## Recovery Procedures

### Database Recovery

**Problem**: Database corrupted or lost

**Solutions**:

```bash
# Stop the agent
sudo systemctl stop research-agent

# Restore from backup
gunzip /opt/research-agent/backups/db_20240114.sql.gz
psql -U research_agent -d knowledge_graph < /opt/research-agent/backups/db_20240114.sql

# Verify data integrity
psql -U research_agent -d knowledge_graph -c "
SELECT COUNT(*) FROM nodes;
SELECT COUNT(*) FROM edges;
"

# Restart agent
sudo systemctl start research-agent
```

### Configuration Recovery

**Problem**: Configuration files corrupted

**Solutions**:

```bash
# Restore from backup
cp /opt/research-agent/backups/config_20240114.tar.gz /tmp/
cd /tmp
tar -xzf config_20240114.tar.gz
cp etc/research-agent.env /etc/
cp opt/research-agent/configs/* /opt/research-agent/configs/

# Reload configuration
sudo systemctl daemon-reload
sudo systemctl restart research-agent
```

### Process Recovery

**Problem**: Agent process crashed

**Solutions**:

```bash
# Check crash logs
sudo journalctl -u research-agent -n 100

# Restart service
sudo systemctl restart research-agent

# Check status
sudo systemctl status research-agent

# Monitor logs
sudo journalctl -u research-agent -f
```

### Data Recovery

**Problem**: Lost research data

**Solutions**:

```sql
-- Check for recent data
SELECT created_at, COUNT(*) as nodes_created
FROM nodes
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY created_at::date
ORDER BY created_at DESC;

-- Restore specific tables if needed
-- (Use backup files to restore individual tables)
```

---

## Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
python3 --version
psql --version

# Agent status
sudo systemctl status research-agent

# Recent logs
tail -50 logs/agent.log
tail -20 logs/ingestion.log

# Database status
psql -U research_agent -d knowledge_graph -c "
SELECT 'nodes' as table_name, COUNT(*) as count FROM nodes
UNION ALL
SELECT 'edges', COUNT(*) FROM edges
UNION ALL
SELECT 'tasks', COUNT(*) FROM research_tasks;
"

# Configuration check
python3 -c "
from research_agent.config import load_config
config = load_config()
print('Configuration loaded successfully')
"
```

### Common Log Messages

**"Connection pool exhausted"**
- Increase database pool size
- Check for connection leaks

**"LLM API quota exceeded"**
- Check API usage limits
- Reduce processing frequency

**"Task timeout"**
- Increase timeout values
- Check network connectivity

**"Memory limit exceeded"**
- Reduce batch sizes
- Increase system memory

### Support Resources

- **Documentation**: Check all sections of this guide
- **Logs**: Comprehensive logging in `logs/` directory
- **Configuration**: Validate YAML files with `python3 -c "import yaml; yaml.safe_load(open('file.yaml'))"`
- **Database**: Direct SQL access for debugging
- **Community**: Check for similar issues in project discussions

---

**Last Updated**: January 14, 2026