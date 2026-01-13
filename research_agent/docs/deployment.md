# Production Deployment Guide

Complete guide for deploying the Autonomous Research Agent in production environments.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Systemd Service Deployment](#systemd-service-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Recovery](#backup-and-recovery)
- [Scaling Considerations](#scaling-considerations)
- [Security Hardening](#security-hardening)

---

## Overview

The Autonomous Research Agent can be deployed in various production environments. This guide covers systemd service deployment, containerization with Docker, and orchestration with Kubernetes.

### Deployment Options

| Method | Use Case | Complexity | Scalability |
|--------|----------|------------|-------------|
| Systemd | Single server | Low | Limited |
| Docker | Containerized | Medium | Moderate |
| Kubernetes | Orchestrated | High | High |
| Cloud | Managed services | Medium | High |

---

## Prerequisites

### System Requirements

- **Linux Server**: Ubuntu 20.04+ or RHEL/CentOS 8+
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD for database and logs
- **Network**: Stable internet connection

### Software Requirements

- **PostgreSQL 14+** with pgvector extension
- **Python 3.10+**
- **Docker** (for containerized deployment)
- **Systemd** (for service management)

### Network Requirements

- **Inbound**: SSH (22), HTTP (80/443) for web UI
- **Outbound**: HTTPS to arXiv, Google APIs, research sources
- **Database**: PostgreSQL port (5432) - can be internal

---

## Systemd Service Deployment

Deploy as a system service on a Linux server.

### Step 1: Prepare Application

```bash
# Create application directory
sudo mkdir -p /opt/research-agent
sudo chown $USER:$USER /opt/research-agent

# Copy application code
cp -r /home/administrator/dev/unbiased-observer/research_agent /opt/research-agent/

# Create virtual environment
cd /opt/research-agent
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r ../knowledge_base/requirements.txt
pip install -r ui/requirements.txt
```

### Step 2: Setup Database

```bash
# Install PostgreSQL
sudo apt update
sudo apt install -y postgresql-14 postgresql-client-14

# Create database and user
sudo -u postgres psql

# In PostgreSQL prompt:
CREATE DATABASE knowledge_graph;
CREATE USER research_agent WITH PASSWORD 'secure_password_here';
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

### Step 3: Configure Environment

```bash
# Create environment file
sudo tee /etc/research-agent.env > /dev/null << 'EOF'
# Required
GOOGLE_API_KEY=your_google_genai_key_here

# Database
DB_CONNECTION_STRING=postgresql://research_agent:secure_password_here@localhost:5432/knowledge_graph

# Optional
DEBUG=0
LOG_LEVEL=INFO
PYTHONPATH=/opt/research-agent
EOF

# Secure permissions
sudo chmod 600 /etc/research-agent.env
sudo chown root:root /etc/research-agent.env
```

### Step 4: Create Systemd Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/research-agent.service > /dev/null << 'EOF'
[Unit]
Description=Autonomous Research Agent
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=research-agent
Group=research-agent
EnvironmentFile=/etc/research-agent.env
WorkingDirectory=/opt/research-agent
ExecStart=/opt/research-agent/venv/bin/python3 research_agent/main.py
Restart=always
RestartSec=10

# Resource limits
MemoryLimit=4G
CPUQuota=400%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=research-agent

[Install]
WantedBy=multi-user.target
EOF
```

### Step 5: Create Service User

```bash
# Create dedicated user
sudo useradd -r -s /bin/false research-agent

# Set permissions
sudo chown -R research-agent:research-agent /opt/research-agent
sudo chown research-agent:research-agent /etc/research-agent.env
```

### Step 6: Configure Logging

```bash
# Create log directories
sudo mkdir -p /var/log/research-agent
sudo chown research-agent:research-agent /var/log/research-agent

# Configure log rotation
sudo tee /etc/logrotate.d/research-agent > /dev/null << 'EOF'
/var/log/research-agent/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 research-agent research-agent
    postrotate
        systemctl reload research-agent
    endscript
}
EOF
```

### Step 7: Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable research-agent

# Start service
sudo systemctl start research-agent

# Check status
sudo systemctl status research-agent

# View logs
sudo journalctl -u research-agent -f
```

### Step 8: Setup Web UI (Optional)

```bash
# Create UI service
sudo tee /etc/systemd/system/research-agent-ui.service > /dev/null << 'EOF'
[Unit]
Description=Research Agent Web UI
After=network.target

[Service]
Type=simple
User=research-agent
Group=research-agent
EnvironmentFile=/etc/research-agent.env
WorkingDirectory=/opt/research-agent/ui
ExecStart=/opt/research-agent/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start UI service
sudo systemctl enable research-agent-ui
sudo systemctl start research-agent-ui
```

### Systemd Management Commands

```bash
# Service control
sudo systemctl start research-agent
sudo systemctl stop research-agent
sudo systemctl restart research-agent
sudo systemctl status research-agent

# View logs
sudo journalctl -u research-agent -f
sudo journalctl -u research-agent --since "1 hour ago"

# Monitor resource usage
sudo systemctl show research-agent | grep -E "(Memory|CPU)"
```

---

## Docker Deployment

Deploy using Docker containers for easier management.

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile for Research Agent
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY knowledge_base/requirements.txt ./knowledge_base/
COPY ui/requirements.txt ./ui/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r knowledge_base/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs cache temp data && \
    chown -R app:app /app

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import psycopg; psycopg.connect('${DB_CONNECTION_STRING}')" || exit 1

# Default command
CMD ["python3", "research_agent/main.py"]
```

### Step 2: Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  database:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: knowledge_graph
      POSTGRES_USER: research_agent
      POSTGRES_PASSWORD: secure_password_here
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U research_agent"]
      interval: 10s
      timeout: 5s
      retries: 5

  research-agent:
    build: .
    environment:
      GOOGLE_API_KEY: your_google_genai_key_here
      DB_CONNECTION_STRING: postgresql://research_agent:secure_password_here@database:5432/knowledge_graph
      DEBUG: 0
      LOG_LEVEL: INFO
    depends_on:
      database:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
    restart: unless-stopped

  ui:
    build: .
    command: streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      GOOGLE_API_KEY: your_google_genai_key_here
      DB_CONNECTION_STRING: postgresql://research_agent:secure_password_here@database:5432/knowledge_graph
    ports:
      - "8501:8501"
    depends_on:
      - research-agent
    restart: unless-stopped

volumes:
  postgres_data:
```

### Step 3: Build and Deploy

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f research-agent
```

### Docker Management

```bash
# Service control
docker-compose start
docker-compose stop
docker-compose restart

# Update deployment
docker-compose pull
docker-compose up -d

# Clean up
docker-compose down -v  # Remove volumes too
docker system prune -a  # Clean unused images
```

---

## Kubernetes Deployment

Deploy on Kubernetes for high availability and scaling.

### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: research-agent
  labels:
    name: research-agent
```

### Step 2: PostgreSQL StatefulSet

```yaml
# postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: research-agent
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg14
        env:
        - name: POSTGRES_DB
          value: "knowledge_graph"
        - name: POSTGRES_USER
          value: "research_agent"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: research-agent-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Step 3: Research Agent Deployment

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-agent
  namespace: research-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: research-agent
  template:
    metadata:
      labels:
        app: research-agent
    spec:
      containers:
      - name: research-agent
        image: your-registry/research-agent:latest
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: research-agent-secrets
              key: google-api-key
        - name: DB_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: research-agent-secrets
              key: db-connection-string
        - name: DEBUG
          value: "0"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
```

### Step 4: Services and Ingress

```yaml
# services.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: research-agent
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432

---
apiVersion: v1
kind: Service
metadata:
  name: research-agent-ui
  namespace: research-agent
spec:
  selector:
    app: research-agent-ui
  ports:
  - port: 8501
    targetPort: 8501
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: research-agent-ingress
  namespace: research-agent
spec:
  rules:
  - host: research-agent.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: research-agent-ui
            port:
              number: 8501
```

### Step 5: Secrets and ConfigMaps

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: research-agent-secrets
  namespace: research-agent
type: Opaque
data:
  google-api-key: <base64-encoded-key>
  db-password: <base64-encoded-password>
  db-connection-string: <base64-encoded-connection-string>

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: research-agent-config
  namespace: research-agent
data:
  config.yaml: |
    llm:
      model_default: "gemini-2.5-flash"
    research:
      arxiv:
        monitoring_enabled: true
```

### Step 6: Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
kubectl apply -f postgres.yaml
kubectl apply -f agent-deployment.yaml
kubectl apply -f services.yaml
kubectl apply -f ingress.yaml

# Check status
kubectl get pods -n research-agent
kubectl get services -n research-agent

# View logs
kubectl logs -f deployment/research-agent -n research-agent
```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance (t3.large or better)
# Ubuntu 20.04 LTS, security group with ports 22, 80, 443

# Install dependencies
sudo apt install -y postgresql-14 postgresql-client-14 python3.10

# Setup application (follow systemd steps above)
```

#### RDS PostgreSQL Setup

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier research-agent-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 14.7 \
    --master-username research_agent \
    --master-user-password your-secure-password \
    --allocated-storage 100 \
    --vpc-security-group-ids sg-your-security-group \
    --db-subnet-group-name your-subnet-group

# Enable pgvector
psql -h your-rds-endpoint -U research_agent -d knowledge_graph -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### S3 Storage (Optional)

```bash
# Create S3 bucket for backups
aws s3 mb s3://research-agent-backups

# Configure backup script
cat > /opt/research-agent/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > /tmp/backup_$DATE.sql
aws s3 cp /tmp/backup_$DATE.sql s3://research-agent-backups/
rm /tmp/backup_$DATE.sql
EOF
```

### Google Cloud Platform

#### Cloud SQL PostgreSQL

```bash
# Create Cloud SQL instance
gcloud sql instances create research-agent-db \
    --database-version=POSTGRES_14 \
    --cpu=2 \
    --memory=8GB \
    --region=us-central1

# Create database
gcloud sql databases create knowledge_graph \
    --instance=research-agent-db

# Create user
gcloud sql users create research_agent \
    --instance=research-agent-db \
    --password=your-secure-password
```

#### GCE VM Setup

```bash
# Create VM instance
gcloud compute instances create research-agent \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --tags=http-server,https-server
```

### Azure Deployment

#### Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres server create \
    --resource-group research-agent-rg \
    --name research-agent-db \
    --location eastus \
    --admin-user research_agent \
    --admin-password your-secure-password \
    --sku-name B_Gen5_2 \
    --storage-size 51200

# Create database
az postgres db create \
    --resource-group research-agent-rg \
    --server-name research-agent-db \
    --name knowledge_graph
```

#### Azure VM Setup

```bash
# Create VM
az vm create \
    --resource-group research-agent-rg \
    --name research-agent-vm \
    --image Ubuntu2204 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --size Standard_D4s_v3
```

---

## Monitoring and Observability

### Application Metrics

```python
# Enable Prometheus metrics (if implemented)
from prometheus_client import start_http_server, Gauge, Counter

# Define metrics
task_counter = Counter('research_agent_tasks_total', 'Total tasks processed', ['type', 'status'])
entity_gauge = Gauge('research_agent_entities_total', 'Total entities in graph')

# Start metrics server
start_http_server(8000)
```

### System Monitoring

#### Prometheus Configuration

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

#### Grafana Dashboard

Create dashboards for:

- Task queue monitoring
- Ingestion pipeline performance
- Community detection metrics
- System resource usage
- LLM API usage statistics

### Log Aggregation

#### ELK Stack Setup

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /var/log/research-agent/*.log
  fields:
    service: research-agent

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### Alerting

#### Alert Manager Rules

```yaml
# alert_rules.yml
groups:
  - name: research-agent
    rules:
      - alert: HighErrorRate
        expr: rate(research_agent_tasks_total{status="failed"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High task failure rate"

      - alert: DatabaseDown
        expr: up{job="research-agent"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Research agent database down"
```

---

## Backup and Recovery

### Database Backup

```bash
# Create backup script
cat > /opt/research-agent/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/research-agent/backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U research_agent -h localhost knowledge_graph > $BACKUP_DIR/db_$DATE.sql

# Compress
gzip $BACKUP_DIR/db_$DATE.sql

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/db_$DATE.sql.gz"
EOF

# Make executable
chmod +x /opt/research-agent/backup.sh
```

### Automated Backups

```bash
# Add to crontab
sudo crontab -e

# Add line for daily backup at 2 AM
0 2 * * * /opt/research-agent/backup.sh
```

### Recovery Procedure

```bash
# Stop services
sudo systemctl stop research-agent

# Restore database
gunzip /opt/research-agent/backups/db_20240114.sql.gz
psql -U research_agent -d knowledge_graph < /opt/research-agent/backups/db_20240114.sql

# Restart services
sudo systemctl start research-agent

# Verify
psql -U research_agent -d knowledge_graph -c "SELECT COUNT(*) FROM nodes;"
```

### Configuration Backup

```bash
# Backup configs
tar -czf /opt/research-agent/backups/config_$(date +%Y%m%d).tar.gz \
    /etc/research-agent.env \
    /opt/research-agent/configs/
```

---

## Scaling Considerations

### Vertical Scaling

```bash
# Increase resources
sudo systemctl stop research-agent

# Edit service file
sudo nano /etc/systemd/system/research-agent.service

# Update limits
MemoryLimit=8G
CPUQuota=800%

sudo systemctl daemon-reload
sudo systemctl start research-agent
```

### Horizontal Scaling

#### Multiple Agent Instances

```bash
# Create multiple services
sudo cp /etc/systemd/system/research-agent.service \
       /etc/systemd/system/research-agent-2.service

# Edit instance-specific settings
sudo nano /etc/systemd/system/research-agent-2.service
# Change working directory, ports, etc.

sudo systemctl enable research-agent-2
sudo systemctl start research-agent-2
```

#### Database Scaling

```bash
# Enable connection pooling
# Install pgbouncer
sudo apt install pgbouncer

# Configure pgbouncer
sudo nano /etc/pgbouncer/pgbouncer.ini

# Add database
[databases]
knowledge_graph = host=localhost port=5432 dbname=knowledge_graph

# Start pgbouncer
sudo systemctl start pgbouncer
```

### Load Balancing

```nginx
# nginx.conf
upstream research_agent_ui {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    server_name research-agent.yourdomain.com;

    location / {
        proxy_pass http://research_agent_ui;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Security Hardening

### Network Security

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8501  # Web UI
sudo ufw allow 5432  # PostgreSQL (restrict to app servers)

# Disable root login
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

### Application Security

```bash
# Run as non-root user
sudo useradd -r -s /bin/false research-agent
sudo chown -R research-agent:research-agent /opt/research-agent

# Secure environment file
sudo chmod 600 /etc/research-agent.env
sudo chown root:research-agent /etc/research-agent.env
```

### Database Security

```sql
-- Create restricted user
CREATE USER research_agent_app WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE knowledge_graph TO research_agent_app;
GRANT USAGE ON SCHEMA public TO research_agent_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO research_agent_app;

-- Revoke unnecessary permissions
REVOKE CREATE ON SCHEMA public FROM research_agent_app;
```

### SSL/TLS Configuration

```nginx
# SSL configuration for web UI
server {
    listen 443 ssl http2;
    server_name research-agent.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Secrets Management

```bash
# Use external secret management
# AWS Secrets Manager, HashiCorp Vault, etc.

# Example with AWS
apt install awscli
aws secretsmanager get-secret-value --secret-id research-agent-secrets
```

### Monitoring Security

```bash
# Secure monitoring endpoints
# Use authentication for Prometheus metrics
# Encrypt log transmission
# Restrict access to monitoring interfaces
```

---

## Performance Benchmarks

### Single Server Performance

- **Task Processing**: 50-100 tasks/hour
- **Content Ingestion**: 10-15 pages/minute
- **Entity Extraction**: 100+ entities/hour
- **Community Detection**: 1000+ nodes in <5 minutes
- **Concurrent Users**: 10-20 for web UI

### Scaling Metrics

- **Database**: 10,000+ entities, 50,000+ relationships
- **Memory**: 2-4GB for basic operation, 8GB+ for large graphs
- **Storage**: 10GB+ for database, logs, and cache
- **Network**: 100MB/day typical, spikes during discovery

### Optimization Tips

1. **Tune PostgreSQL**: Adjust shared_buffers, work_mem, maintenance_work_mem
2. **Monitor Resources**: Use htop, iotop, pg_top for performance monitoring
3. **Cache Results**: Implement caching for expensive operations
4. **Batch Operations**: Process multiple items together
5. **Async Processing**: Maximize concurrent operations

---

## Troubleshooting Production Issues

### Service Startup Issues

```bash
# Check service status
sudo systemctl status research-agent

# View detailed logs
sudo journalctl -u research-agent -n 50

# Check environment
sudo -u research-agent bash -c 'source /etc/research-agent.env && env | grep -E "(GOOGLE|DB)"'
```

### Database Connection Issues

```bash
# Test connection
psql -U research_agent -h localhost -d knowledge_graph -c "SELECT 1;"

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log

# Verify pgvector
psql -U research_agent -d knowledge_graph -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Performance Issues

```bash
# Monitor CPU usage
top -p $(pgrep -f research_agent)

# Check database performance
psql -U research_agent -d knowledge_graph -c "SELECT * FROM pg_stat_activity;"

# Analyze slow queries
psql -U research_agent -d knowledge_graph -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"
```

### Memory Issues

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Adjust systemd limits
sudo nano /etc/systemd/system/research-agent.service
# Add: MemoryLimit=4G
sudo systemctl daemon-reload
sudo systemctl restart research-agent
```

For more troubleshooting tips, see [Troubleshooting Guide](troubleshooting.md).

---

**Last Updated**: January 14, 2026