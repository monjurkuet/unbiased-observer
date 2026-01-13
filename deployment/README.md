# Deployment Guide - Autonomous Research Agent

## Overview

This guide covers production deployment of the Autonomous Research Agent, a 24/7 AI-powered research assistant that automatically discovers, processes, and organizes research content.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Python**: 3.10 or higher
- **PostgreSQL**: 13+ with pgvector extension
- **Memory**: 8GB+ RAM recommended
- **Storage**: 50GB+ for knowledge graphs and logs
- **Network**: Stable internet connection for arXiv API access

### External Services
- **LLM API**: Local LLM server (LM Studio, Ollama) on `http://localhost:8317/v1`
- **Google GenAI**: API key for embeddings
- **PostgreSQL**: Database server with pgvector

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Database Setup

```bash
# Create database user
sudo -u postgres createuser --createdb --login agentzero
sudo -u postgres psql -c "ALTER USER agentzero PASSWORD 'secure_password';"

# Create database
sudo -u postgres createdb knowledge_graph -O agentzero

# Enable extensions
sudo -u postgres psql knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
sudo -u postgres psql knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"

# Run schema
sudo -u postgres psql knowledge_graph < research_agent/db_schema.sql
```

### 3. Application Setup

```bash
# Clone repository
git clone <repository-url>
cd unbiased-observer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs cache state configs

# Copy configuration templates
cp configs/research_agent_config.yaml.example configs/research_agent_config.yaml
cp configs/research_sources.yaml.example configs/research_sources.yaml
```

### 4. Configuration

Edit `configs/research_agent_config.yaml`:

```yaml
database:
  connection_string: "postgresql://agentzero:secure_password@localhost:5432/knowledge_graph"

llm:
  base_url: "http://localhost:8317/v1"
  api_key: "lm-studio"

# ... other settings
```

Set environment variables:

```bash
export GOOGLE_API_KEY="your_google_genai_key"
echo "export GOOGLE_API_KEY='your_google_genai_key'" >> ~/.bashrc
```

## Production Deployment

### Option 1: Systemd Service (Recommended)

```bash
# Create system user
sudo useradd -r -s /bin/false research-agent

# Set permissions
sudo chown -R research-agent:research-agent /home/administrator/dev/unbiased-observer

# Copy service file
sudo cp deployment/research-agent.service /etc/systemd/system/

# Edit service file with correct paths and credentials
sudo nano /etc/systemd/system/research-agent.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable research-agent
sudo systemctl start research-agent

# Check status
sudo systemctl status research-agent
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV GOOGLE_API_KEY=your_key_here

CMD ["python3", "research_agent/main.py"]
```

```bash
# Build and run
docker build -t research-agent .
docker run -d --name research-agent \
  -v /path/to/configs:/app/configs \
  -v /path/to/logs:/app/logs \
  --network host \
  research-agent
```

### Option 3: Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'research-agent',
    script: 'research_agent/main.py',
    cwd: '/home/administrator/dev/unbiased-observer',
    env: {
      PYTHONPATH: '/home/administrator/dev/unbiased-observer',
      GOOGLE_API_KEY: 'your_key_here'
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '4G'
  }]
}
EOF

# Start with PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## Monitoring & Maintenance

### Log Monitoring

```bash
# View real-time logs
tail -f logs/agent.log
tail -f logs/ingestion.log
tail -f logs/processing.log

# Search for errors
grep "ERROR" logs/*.log
grep "FAILED" logs/*.log
```

### Database Monitoring

```sql
-- Check system health
SELECT
  (SELECT COUNT(*) FROM research_tasks WHERE status = 'PENDING') as pending_tasks,
  (SELECT COUNT(*) FROM research_tasks WHERE status = 'FAILED') as failed_tasks,
  (SELECT COUNT(*) FROM nodes) as entities,
  (SELECT COUNT(*) FROM edges) as relationships;

-- Recent activity
SELECT type, status, created_at, error_message
FROM research_tasks
ORDER BY created_at DESC LIMIT 20;

-- Performance metrics
SELECT
  AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_time,
  COUNT(*) as total_processed
FROM research_tasks
WHERE status = 'COMPLETED' AND completed_at > NOW() - INTERVAL '1 hour';
```

### Health Checks

```bash
# Quick health check
python3 -c "
import asyncio
from research_agent.config import load_config
from research_agent.orchestrator import TaskQueue

async def health_check():
    config = load_config()
    task_queue = TaskQueue(config.database.connection_string)
    await task_queue.initialize()
    count = await task_queue.get_pending_count()
    print(f'✅ Agent healthy - {count} pending tasks')

asyncio.run(health_check())
"
```

### Backup Strategy

```bash
# Database backup
pg_dump knowledge_graph > backup_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz configs/

# Log rotation (logrotate)
cat > /etc/logrotate.d/research-agent << EOF
/home/administrator/dev/unbiased-observer/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
EOF
```

## Scaling & Performance

### Horizontal Scaling

```bash
# Run multiple instances
for i in {1..3}; do
  cp deployment/research-agent.service deployment/research-agent-$i.service
  sed -i "s/research-agent/research-agent-$i/g" deployment/research-agent-$i.service
  sudo cp deployment/research-agent-$i.service /etc/systemd/system/
  sudo systemctl enable research-agent-$i
  sudo systemctl start research-agent-$i
done
```

### Performance Tuning

```yaml
# configs/research_agent_config.yaml
research:
  max_concurrent_fetches: 20      # Increase for more throughput
  rate_limit: 5.0                 # API rate limiting

ingestion:
  max_concurrent_llm_calls: 10    # Parallel LLM processing
  batch_size: 50                  # Larger batches

processing:
  min_entities_to_process: 500    # Process more frequently
```

### Resource Monitoring

```bash
# CPU and memory usage
top -p $(pgrep -f "research_agent/main.py")

# Database connections
psql knowledge_graph -c "SELECT count(*) FROM pg_stat_activity;"

# Disk usage
du -sh logs/ cache/ state/
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql postgresql://agentzero:password@localhost:5432/knowledge_graph -c "SELECT 1;"

# Check connection string in config
grep "connection_string" configs/research_agent_config.yaml
```

#### LLM API Issues
```bash
# Test LLM API
curl http://localhost:8317/v1/models

# Check API logs
tail -f /path/to/llm/server/logs
```

#### Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Adjust memory limits in systemd service
sudo nano /etc/systemd/system/research-agent.service
# Add: MemoryLimit=8G
sudo systemctl daemon-reload
sudo systemctl restart research-agent
```

#### High CPU Usage
```bash
# Check what's consuming CPU
top -p $(pgrep -f "research_agent/main.py")

# Reduce concurrency in config
# Lower max_concurrent_fetches and max_concurrent_llm_calls
```

### Recovery Procedures

#### Restart Agent
```bash
sudo systemctl restart research-agent
# or
pm2 restart research-agent
```

#### Clear Failed Tasks
```sql
-- Reset stuck tasks (use carefully)
UPDATE research_tasks
SET status = 'PENDING', worker_id = NULL, started_at = NULL, retry_count = 0
WHERE status = 'IN_PROGRESS'
  AND started_at < NOW() - INTERVAL '1 hour';
```

#### Rebuild Knowledge Graph
```sql
-- Clear and rebuild (destructive!)
TRUNCATE nodes, edges, events, communities CASCADE;
-- Then restart agent to rebuild from sources
```

## Security Considerations

### API Key Management
```bash
# Store keys securely
echo "GOOGLE_API_KEY=your_key" | sudo tee /etc/research-agent.env
sudo chmod 600 /etc/research-agent.env

# Reference in systemd service
EnvironmentFile=/etc/research-agent.env
```

### Database Security
```sql
-- Create restricted user
CREATE USER agent_readonly WITH PASSWORD 'readonly_pass';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_readonly;

-- Use for monitoring dashboards
```

### Network Security
```bash
# Restrict agent network access
sudo ufw allow from trusted_ip to any port 8317  # LLM API
sudo ufw allow from trusted_ip to any port 5432  # PostgreSQL

# Use VPN for remote access
```

## Support & Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor logs for errors
- Check task queue depth
- Verify database connectivity

**Weekly:**
- Review failed tasks and error patterns
- Analyze research coverage effectiveness
- Update arXiv search keywords

**Monthly:**
- Performance analysis and optimization
- Dependency updates and security patches
- Backup verification

### Getting Help

- **Logs**: Check `logs/agent.log` for detailed error information
- **Database**: Query `research_tasks` and `ingestion_logs` for debugging
- **Configuration**: Validate YAML syntax and paths
- **Dependencies**: Ensure all packages in `requirements.txt` are installed

---

**For additional support, check the project documentation or create an issue in the repository.**