# Installation Guide

Complete setup instructions for the Autonomous Research Agent system.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Step 1: Clone Repository](#step-1-clone-repository)
  - [Step 2: Install System Dependencies](#step-2-install-system-dependencies)
  - [Step 3: Setup PostgreSQL](#step-3-setup-postgresql)
  - [Step 4: Install Python Dependencies](#step-4-install-python-dependencies)
  - [Step 5: Configure Environment](#step-5-configure-environment)
  - [Step 6: Initialize Database](#step-6-initialize-database)
  - [Step 7: Verify Installation](#step-7-verify-installation)
- [Installation Options](#installation-options)
  - [Development Mode](#development-mode)
  - [Production Mode](#production-mode)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL2 recommended)
- **Python**: 3.10 or higher
- **PostgreSQL**: 14 or higher with `pgvector` extension
- **RAM**: Minimum 4GB (8GB recommended for production)
- **Disk Space**: 10GB minimum for database and logs
- **Network**: Internet connection for arXiv API and LLM services

### Software Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Core runtime |
| PostgreSQL | 14+ | Knowledge graph storage |
| pgvector | 0.4.0+ | Vector similarity search |
| pip | Latest | Package manager |
| Git | Latest | Version control |

---

## Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd unbiased-observer/research_agent

# Verify repository structure
ls -la
```

### Step 2: Install System Dependencies

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL 14
sudo apt install -y postgresql-14 postgresql-client-14

# Install Python development headers
sudo apt install -y python3.10-dev python3-pip

# Install system dependencies for Python packages
sudo apt install -y build-essential libpq-dev
```

#### macOS

```bash
# Install PostgreSQL using Homebrew
brew install postgresql@14

# Start PostgreSQL service
brew services start postgresql@14

# Install Python 3.10 (if not already installed)
brew install python@3.10
```

#### Windows (WSL2)

```bash
# Install PostgreSQL in WSL2
sudo apt update
sudo apt install -y postgresql-14 postgresql-client-14 python3.10-dev python3-pip
```

### Step 3: Setup PostgreSQL

#### Install pgvector Extension

```bash
# Install pgvector for PostgreSQL
# For Ubuntu/Debian:
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
# Or use: sudo apt install postgresql-14-pgvector (if available)

# For macOS:
brew install pgvector
```

#### Create Database

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL prompt:
CREATE DATABASE knowledge_graph;
CREATE USER agentzero WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE knowledge_graph TO agentzero;
ALTER USER agentzero WITH SUPERUSER;  # Required for pgvector extension
\q
```

#### Verify pgvector

```bash
# Connect to database
psql -U postgres -d knowledge_graph

# In PostgreSQL prompt:
CREATE EXTENSION IF NOT EXISTS vector;
\dx
# Verify pgvector is listed
\q
```

### Step 4: Install Python Dependencies

#### Main Agent Dependencies

```bash
# Set PYTHONPATH
export PYTHONPATH=/home/administrator/dev/unbiased-observer

# Install main requirements
pip install -r requirements.txt
```

#### Knowledge Base Dependencies

```bash
# Install knowledge base requirements
pip install -r ../knowledge_base/requirements.txt
```

#### UI Dependencies

```bash
# Install UI requirements
pip install -r ui/requirements.txt
```

#### Verify Installation

```bash
# Verify critical packages
python3 -c "import asyncio; print('asyncio: OK')"
python3 -c "import psycopg; print('psycopg: OK')"
python3 -c "import google.generativeai; print('google.generativeai: OK')"
python3 -c "import apscheduler; print('apscheduler: OK')"
python3 -c "import aiohttp; print('aiohttp: OK')"
```

### Step 5: Configure Environment

#### Set Environment Variables

```bash
# Create .env file in project root
cat > .env << 'EOF'
# Required
GOOGLE_API_KEY=your_google_genai_key_here

# Database (optional if using config.yaml)
DB_CONNECTION_STRING=postgresql://agentzero:your_password@localhost:5432/knowledge_graph

# LLM Configuration (optional)
LLM_BASE_URL=http://localhost:8317/v1
LLM_API_KEY=lm-studio

# Debug/Logging (optional)
DEBUG=1
LOG_LEVEL=INFO
EOF

# Source environment variables
source .env
```

#### Alternative: Export Directly

```bash
# Set required API key
export GOOGLE_API_KEY="your_google_genai_key_here"

# Optional settings
export DEBUG=1
export LOG_LEVEL=INFO
```

### Step 6: Initialize Database

```bash
# Load database schema
psql -U postgres -d knowledge_graph < db_schema.sql

# Verify tables created
psql -U postgres -d knowledge_graph -c "\dt"

# Expected tables:
# - nodes
# - edges
# - events
# - communities
# - community_membership
# - community_hierarchy
# - research_tasks
# - research_sources
# - ingestion_logs
# - processing_logs
# - agent_state
```

### Step 7: Verify Installation

#### Test Core Components

```bash
# Test 1: Configuration loading
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
```

#### Test arXiv Integration

```bash
# Test 2: arXiv API integration
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
from research_agent.research import ArxivIntegrator
integrator = ArxivIntegrator()
papers = integrator.search_by_keywords(['machine learning'], max_results=3)
print(f'✅ Found {len(papers)} papers from arXiv')
"
```

#### Test Database Connection

```bash
# Test 3: Database connection
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 -c "
import asyncio
from psycopg import AsyncConnection

async def test():
    conn_str = 'postgresql://agentzero@localhost:5432/knowledge_graph'
    async with await AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT 1')
            result = await cur.fetchone()
            print(f'✅ Database connection successful: {result[0]}')

asyncio.run(test())
"
```

If all tests pass, your installation is complete!

---

## Installation Options

### Development Mode

For development and testing:

```bash
# Enable debug mode
export DEBUG=1
export LOG_LEVEL=DEBUG

# Run agent with verbose logging
PYTHONPATH=/home/administrator/dev/unbiased-observer python3 research_agent/main.py
```

**Development Features**:
- Debug-level logging
- Detailed error traces
- Database query logging
- Task queue debugging
- Performance metrics

### Production Mode

For production deployment:

```bash
# Use production settings
export LOG_LEVEL=INFO
export DEBUG=0

# Run with systemd service (see deployment guide)
sudo systemctl start research-agent
sudo systemctl enable research-agent
```

**Production Features**:
- Optimized logging (INFO level)
- Structured JSON logging
- Process supervision
- Automatic restart
- Health monitoring

For production deployment details, see [Deployment Guide](deployment.md).

---

## Configuration Files

After installation, you may need to configure the agent:

### Main Configuration

```bash
# Copy example configuration (if available)
cp configs/research_agent_config.yaml.example configs/research_agent_config.yaml

# Edit configuration
nano configs/research_agent_config.yaml
```

### Research Sources Configuration

```bash
# Copy example sources (if available)
cp configs/research_sources.yaml.example configs/research_sources.yaml

# Edit sources
nano configs/research_sources.yaml
```

For complete configuration reference, see [Configuration Guide](configuration.md).

---

## Upgrading

### Upgrading Dependencies

```bash
# Update requirements
pip install --upgrade -r requirements.txt

# Freeze to lock file
pip freeze > requirements-lock.txt
```

### Database Migrations

```bash
# Backup current database
pg_dump -U postgres knowledge_graph > backup_$(date +%Y%m%d).sql

# Apply new schema (if applicable)
psql -U postgres -d knowledge_graph < db_schema_v2.sql
```

---

## Uninstallation

```bash
# Stop services
sudo systemctl stop research-agent  # If running as service

# Remove Python packages
pip uninstall -y -r requirements.txt

# Drop database
sudo -u postgres psql -c "DROP DATABASE knowledge_graph;"
sudo -u postgres psql -c "DROP USER agentzero;"

# Remove repository
cd ../..
rm -rf unbiased-observer
```

---

## Troubleshooting

### PostgreSQL Issues

**Problem**: Connection refused
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Start PostgreSQL if not running
sudo systemctl start postgresql
```

**Problem**: pgvector extension not found
```bash
# Install pgvector
sudo apt install postgresql-14-pgvector

# Or compile from source (see Step 3)
```

### Python Issues

**Problem**: Module not found
```bash
# Set PYTHONPATH
export PYTHONPATH=/home/administrator/dev/unbiased-observer

# Add to .bashrc for persistence
echo 'export PYTHONPATH=/home/administrator/dev/unbiased-observer' >> ~/.bashrc
source ~/.bashrc
```

**Problem**: pip installation fails
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Database Connection Issues

**Problem**: FATAL: password authentication failed
```bash
# Verify database credentials
psql -U postgres -d knowledge_graph -c "SELECT usename FROM pg_user;"

# Reset password if needed
sudo -u postgres psql -c "ALTER USER agentzero WITH PASSWORD 'new_password';"
```

**Problem**: FATAL: database does not exist
```bash
# Create database
sudo -u postgres createdb knowledge_graph

# Load schema
psql -U postgres -d knowledge_graph < db_schema.sql
```

### API Key Issues

**Problem**: GOOGLE_API_KEY not set
```bash
# Set API key
export GOOGLE_API_KEY="your_key_here"

# Add to .env file
echo "GOOGLE_API_KEY=your_key_here" >> .env
source .env
```

For more troubleshooting tips, see [Troubleshooting Guide](troubleshooting.md).

---

## Next Steps

After successful installation:

1. **Configure the agent**: See [Configuration Guide](configuration.md)
2. **Start the agent**: `PYTHONPATH=/home/administrator/dev/unbiased-observer python3 research_agent/main.py`
3. **Access the UI**: `cd ui && python run.py` → http://localhost:8501
4. **Deploy to production**: See [Deployment Guide](deployment.md)
5. **Explore examples**: See [Examples](examples/)

---

## Support

If you encounter issues during installation:

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review logs in `./logs/` directory
- Open an issue on GitHub with error details
- Join community discussions for help

---

**Last Updated**: January 14, 2026
