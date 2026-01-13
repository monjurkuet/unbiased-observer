# Quick Start Guide

Step-by-step tutorial for getting started with the Autonomous Research Agent.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [First Run](#first-run)
- [Adding Research Content](#adding-research-content)
- [Monitoring the Agent](#monitoring-the-agent)
- [Querying the Knowledge Graph](#querying-the-knowledge-graph)
- [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **PostgreSQL 14+** with `pgvector` extension
- **Google GenAI API key** (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- **4GB RAM minimum** (8GB recommended)
- **Internet connection** for arXiv API and LLM services

### System Check

```bash
# Check Python version
python3 --version
# Should show: Python 3.10.x or higher

# Check PostgreSQL (if installed)
psql --version
# Should show: psql (PostgreSQL) 14.x or higher
```

---

## Installation

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd unbiased-observer/research_agent

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set PYTHONPATH
export PYTHONPATH=/home/administrator/dev/unbiased-observer
```

### Step 2: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install knowledge base dependencies
pip install -r ../knowledge_base/requirements.txt

# Install UI dependencies
pip install -r ui/requirements.txt
```

### Step 3: Setup PostgreSQL Database

```bash
# Start PostgreSQL service (if not running)
sudo systemctl start postgresql  # Linux
# brew services start postgresql  # macOS

# Create database and user
sudo -u postgres psql

# In PostgreSQL prompt:
CREATE DATABASE knowledge_graph;
CREATE USER research_agent WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE knowledge_graph TO research_agent;
ALTER USER research_agent WITH SUPERUSER;
\q

# Install pgvector
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable pgvector
sudo -u postgres psql -d knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 4: Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
# Required: Get from https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_genai_key_here

# Database connection
DB_CONNECTION_STRING=postgresql://research_agent:your_password@localhost:5432/knowledge_graph

# Optional settings
DEBUG=1
LOG_LEVEL=INFO
EOF

# Load environment
source .env
```

### Step 5: Initialize Database Schema

```bash
# Load database schema
psql -U postgres -d knowledge_graph < db_schema.sql

# Verify tables created
psql -U postgres -d knowledge_graph -c "\dt"
# Should show: nodes, edges, events, communities, etc.
```

---

## Configuration

### Basic Configuration

Create the main configuration file:

```bash
# Copy example configuration
cp configs/research_agent_config.yaml.example configs/research_agent_config.yaml

# Edit basic settings
nano configs/research_agent_config.yaml
```

### Essential Settings

```yaml
# Database connection
database:
  connection_string: "postgresql://research_agent:password@localhost:5432/knowledge_graph"

# LLM settings
llm:
  base_url: "http://localhost:8317/v1"  # Or your LLM server
  api_key: "lm-studio"
  model_default: "gemini-2.5-flash"

# Research monitoring
research:
  arxiv:
    monitoring_enabled: true
    monitoring_interval_hours: 2
```

### Research Sources Configuration

```bash
# Copy sources configuration
cp configs/research_sources.yaml.example configs/research_sources.yaml

# Configure arXiv monitoring
nano configs/research_sources.yaml
```

```yaml
sources:
  - type: "arxiv"
    name: "AI Research"
    config:
      keywords: ["artificial intelligence", "machine learning"]
      max_results: 5
      days_back: 7
    active: true
    priority: "high"
```

---

## First Run

### Test Installation

```bash
# Test core components
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
import asyncio
from research_agent.config import load_config
from research_agent.monitoring import setup_logging

async def test():
    config = load_config()
    agent_logger, ingestion_logger, processing_logger, orchestrator_logger = setup_logging(config, debug=True)
    print('✅ All components initialized successfully')

asyncio.run(test())
"

# Test arXiv integration
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
from research_agent.research import ArxivIntegrator
integrator = ArxivIntegrator()
papers = integrator.search_by_keywords(['machine learning'], max_results=3)
print(f'✅ Found {len(papers)} papers from arXiv')
"
```

### Start the Agent

```bash
# Start the autonomous research agent
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 research_agent/main.py

# Expected output:
# 2024-01-14 10:30:00,123 - agent - INFO - Autonomous Research Agent starting...
# 2024-01-14 10:30:00,234 - orchestrator - INFO - Scheduler initialized
# 2024-01-14 10:30:00,345 - agent - INFO - Agent started successfully
```

The agent will now run continuously, monitoring arXiv every 2 hours and processing any research content it discovers.

### Check Initial Status

```bash
# Monitor logs
tail -f logs/agent.log

# Check database status
psql -U postgres -d knowledge_graph -c "SELECT status, COUNT(*) FROM research_tasks GROUP BY status;"

# Check system health
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
from research_agent.monitoring import MetricsCollector
import asyncio

async def check():
    metrics = MetricsCollector(config)
    summary = await metrics.get_summary_metrics()
    print(f'System health: {summary[\"system_health\"][\"database_connected\"]}')
    print(f'Pending tasks: {summary[\"task_metrics\"][\"pending_tasks\"]}')

asyncio.run(check())
"
```

---

## Adding Research Content

### Method 1: Automatic arXiv Monitoring

The agent automatically monitors arXiv based on your configuration. It will:

1. Search for papers matching your keywords
2. Download new papers every 2 hours
3. Extract entities and relationships
4. Build the knowledge graph

Check that papers are being discovered:

```bash
# View recent tasks
psql -U postgres -d knowledge_graph -c "
SELECT id, task_type, status, created_at
FROM research_tasks
ORDER BY created_at DESC
LIMIT 5;
"
```

### Method 2: Manual Content Addition

Add specific research content manually:

```python
from research_agent.research import ManualSourceManager
from research_agent.orchestrator import TaskQueue
import asyncio

async def add_content():
    # Initialize components
    config = load_config()
    queue = TaskQueue(config)
    await queue.initialize()
    
    manager = ManualSourceManager(queue)
    
    # Add different types of content
    examples = [
        {
            "type": "url",
            "url": "https://arxiv.org/pdf/2301.07041.pdf",
            "metadata": {"title": "Attention Is All You Need", "year": 2023}
        },
        {
            "type": "text",
            "text": """
            Recent advances in transformer architectures have revolutionized
            natural language processing. The key innovation is the multi-head
            attention mechanism that allows the model to attend to different
            parts of the input simultaneously.
            """,
            "metadata": {"topic": "transformers", "source": "notes"}
        }
    ]

    for example in examples:
        if example["type"] == "url":
            task_id = await manager.add_url_source(
                url=example["url"],
                metadata=example["metadata"]
            )
            print(f"Added URL source: {example['url']} (Task ID: {task_id})")
        elif example["type"] == "text":
            task_id = await manager.add_text_source(
                text=example["text"],
                metadata=example["metadata"]
            )
            print(f"Added text content (Task ID: {task_id})")

    await queue.close()
    print("Content addition complete!\n")

asyncio.run(add_content())
```

### Method 3: File Upload

Add local research files:

```python
# Add PDF file
task_id = await manager.add_file_source(
    file_path="/path/to/research_paper.pdf",
    metadata={"author": "Dr. Smith", "conference": "ICML 2023"}
)
```

---

## Monitoring the Agent

### View System Status

```bash
# Check agent logs
tail -f logs/agent.log

# View ingestion progress
tail -f logs/ingestion.log

# Monitor task queue
watch -n 5 "psql -U postgres -d knowledge_graph -c 'SELECT status, COUNT(*) FROM research_tasks GROUP BY status;'"
```

### Real-time Metrics

```python
from research_agent.monitoring import MetricsCollector
import asyncio

async def show_metrics():
    metrics = MetricsCollector(config)
    summary = await metrics.get_summary_metrics()
    
    print("=== System Metrics ===")
    print(f"Total entities: {summary['ingestion_metrics']['entities_extracted']:,}")
    print(f"Total relationships: {summary['ingestion_metrics']['relationships_extracted']:,}")
    print(f"Communities found: {summary['processing_metrics']['total_communities']}")

    # Task queue status
    task_metrics = summary['task_metrics']
    print("
Task Queue:")
    print(f"  Total tasks: {task_metrics['total_tasks']}")
    print(f"  Pending: {task_metrics['pending_tasks']}")
    print(f"  Completed: {task_metrics['completed_tasks']}")
    print(f"  Failed: {task_metrics['failed_tasks']}")
    print(".1f")

    print("System monitoring complete!\n")

asyncio.run(show_metrics())
```

### Web Interface

Start the web interface to visualize the knowledge graph:

```bash
# Start UI
cd ui
python run.py

# Open browser to http://localhost:8501
```

The interface shows:
- **Dashboard**: Real-time statistics
- **Knowledge Graph**: Interactive network visualization
- **Query Interface**: Natural language questions
- **Analytics**: Charts and trends

---

## Querying the Knowledge Graph

### Basic Database Queries

```sql
-- View entities
SELECT id, name, type, description FROM nodes LIMIT 10;

-- View relationships
SELECT n1.name as source, e.type, n2.name as target
FROM edges e
JOIN nodes n1 ON e.source_id = n1.id
JOIN nodes n2 ON e.target_id = n2.id
LIMIT 10;

-- View communities
SELECT name, size, summary FROM communities ORDER BY size DESC;

-- Recent activity
SELECT task_type, status, created_at FROM research_tasks
ORDER BY created_at DESC LIMIT 5;
```

### Advanced Queries

```sql
-- Find papers by author
SELECT n1.name as paper, n2.name as author
FROM edges e
JOIN nodes n1 ON e.source_id = n1.id
JOIN nodes n2 ON e.target_id = n2.id
WHERE e.type = 'authored_by' AND n2.name LIKE '%Vaswani%';

-- Find related concepts
SELECT n2.name, COUNT(*) as connections
FROM edges e
JOIN nodes n1 ON e.source_id = n1.id
JOIN nodes n2 ON e.target_id = n2.id
WHERE n1.name = 'Transformer Architecture' AND n2.type = 'concept'
GROUP BY n2.name
ORDER BY connections DESC;

-- Community analysis
SELECT c.name, c.size, c.summary, COUNT(cm.node_id) as members
FROM communities c
LEFT JOIN community_membership cm ON c.id = cm.community_id
GROUP BY c.id, c.name, c.size, c.summary
ORDER BY c.size DESC;
```

### Semantic Search

```python
# Search by meaning using vector embeddings
from research_agent.ingestion import DirectPostgresStorage
import asyncio

async def semantic_search():
    storage = DirectPostgresStorage(config)
    await storage.initialize()
    
    # This would require vector search implementation
    # For now, use text matching
    query = "attention mechanism"
    
    # Placeholder for vector search
    print(f"Searching for: {query}")
    
    await storage.close()

asyncio.run(semantic_search())
```

---

## Troubleshooting Common Issues

### Agent Won't Start

```bash
# Check environment variables
echo $GOOGLE_API_KEY
echo $PYTHONPATH

# Test imports
python3 -c "from research_agent.config import load_config; print('Config OK')"

# Check database connection
psql -U postgres -d knowledge_graph -c "SELECT 1;"
```

### No Tasks Being Processed

```bash
# Check scheduler status
ps aux | grep research_agent

# View task queue
psql -U postgres -d knowledge_graph -c "SELECT * FROM research_tasks LIMIT 5;"

# Check for errors
tail -20 logs/orchestrator.log
```

### LLM API Errors

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

### Database Issues

```bash
# Check connection
psql -U research_agent -d knowledge_graph -c "SELECT version();"

# Verify schema
psql -U research_agent -d knowledge_graph -c "\dt"

# Check permissions
psql -U research_agent -d knowledge_graph -c "SELECT current_user;"
```

---

## Next Steps

### Explore Advanced Features

1. **Customize Research Sources**
   - Add RSS feeds, custom APIs, or file directories
   - Configure keyword monitoring for your field
   - Set up priority-based processing

2. **Extend the Knowledge Graph**
   - Add custom entity types and relationships
   - Implement domain-specific extraction rules
   - Integrate with external knowledge bases

3. **Optimize Performance**
   - Tune LLM models and batch sizes
   - Configure processing triggers
   - Set up monitoring alerts

### Production Deployment

1. **Systemd Service Setup**
   ```bash
   sudo cp deployment/research-agent.service /etc/systemd/system/
   sudo systemctl enable research-agent
   sudo systemctl start research-agent
   ```

2. **Monitoring Setup**
   - Configure log rotation
   - Set up health checks
   - Enable performance monitoring

3. **Backup Strategy**
   - Schedule database backups
   - Archive processed content
   - Document recovery procedures

### What's Next

- **Processing Pipeline**: The agent will automatically detect research communities
- **Web Interface**: Full visualization of your knowledge graph
- **Query Capabilities**: Natural language questions about research
- **Analytics**: Trends and insights from your research data

Congratulations! You now have a running Autonomous Research Agent that will continuously build and analyze a knowledge graph of research in your field.

---

**Last Updated**: January 14, 2026