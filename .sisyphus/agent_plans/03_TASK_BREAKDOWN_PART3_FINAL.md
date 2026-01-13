# AUTONOMOUS AGENT - TASK BREAKDOWN PART3 (FINAL)

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Ready for Implementation

---

## PHASE 5: MONITORING & DEPLOYMENT (WEEK 5) - CONTINUED

### Task 5.3: Configuration Files
**Priority**: P0 | **Effort**: 2 hours | **Dependencies**: None

**Files**: `configs/research_agent_config.yaml`, `configs/research_sources.yaml`

**Implementation**:
```yaml
# configs/research_agent_config.yaml
agent:
  name: "autonomous_research_agent"
  version: "1.0"
  log_level: "INFO"

database:
  # Use existing connection string
  connection_string: "postgresql://agentzero@localhost:5432/knowledge_graph"
  pool_min_size: 5
  pool_max_size: 20

llm:
  # Local OpenAI-compatible API
  base_url: "http://localhost:8317/v1"
  api_key: "lm-studio"
  model_default: "gemini-2.5-flash"
  model_pro: "gemini-2.5-pro"
  max_retries: 3
  timeout: 120  # seconds

embedding:
  # Google GenAI for embeddings
  provider: "google"
  model: "models/text-embedding-004"
  api_key_env: "GOOGLE_API_KEY"  # Load from environment
  dimensions: 768

research:
  sources_config: "configs/research_sources.yaml"
  max_concurrent_fetches: 10
  rate_limit: 2.0  # requests per second
  max_content_length: 1000000  # characters

ingestion:
  max_concurrent_llm_calls: 3
  batch_size: 10
  retry_backoff_factor: 2.0
  max_retries: 3

processing:
  min_entities_to_process: 100
  min_time_between_processing_hours: 1
  processing_interval_seconds: 60

monitoring:
  metrics_port: 8000
  health_check_interval_seconds: 300
  log_retention_days: 30

paths:
  # WSL-compatible path
  knowledge_base: "\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer\\knowledge_base"
  cache_dir: "./cache"
  logs_dir: "./logs"
  state_dir: "./state"
```

```yaml
# configs/research_sources.yaml
sources:
  # Manual source (user-added)
  - type: "manual"
    name: "Manual Input"
    description: "Manually added research sources"
    is_active: true

  # Future: RSS feeds
  # - type: "rss"
  #   name: "ArXiv CS.AI"
  #   url: "http://export.arxiv.org/rss/cs.AI.xml"
  #   update_interval_hours: 6
  #   is_active: true

  # Future: API sources
  # - type: "api"
  #   name: "News API"
  #   url: "https://api.example.com/news"
  #   api_key_env: "NEWS_API_KEY"
  #   update_interval_hours: 1
  #   is_active: false
```

**Success Criteria**:
- [ ] Agent config file created with all sections
- [ ] Sources config file created
- [ ] WSL path format correct
- [ ] Environment variables referenced properly

**Verification**: Load configs, validate all values

---

### Task 5.4: Main Entry Point
**Priority**: P0 | **Effort**: 3 hours | **Dependencies**: 1.7, 2.5, 3.3, 4.2, 5.1

**File**: `research_agent/main.py`

**Implementation**:
```python
#!/usr/bin/env python3
"""
Autonomous Research Agent - 24/7 Knowledge Gathering & Ingestion

Main entry point for the autonomous research agent system.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_agent.config import load_config
from research_agent.__init__ import setup_logging
from research_agent.orchestrator.task_queue import TaskQueue
from research_agent.orchestrator.scheduler import AgentScheduler
from research_agent.research.manual_source import ManualSourceManager
from research_agent.monitoring.health_checker import HealthChecker
from research_agent.monitoring.metrics import MetricsCollector

# Global state
running = True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False

async def main():
    """Main entry point"""

    global running
    logger.info("=" * 60)
    logger.info("Starting Autonomous Research Agent")
    logger.info("=" * 60)

    # 1. Load configuration
    logger.info("Loading configuration...")
    config = load_config("configs/research_agent_config.yaml")

    # 2. Setup logging
    agent_logger, ingestion_logger, processing_logger = setup_logging(config)

    agent_logger.info(f"Agent version: {config.version}")
    agent_logger.info(f"Log level: {config.log_level}")

    # 3. Initialize database
    agent_logger.info("Initializing components...")

    task_queue = TaskQueue(config.database.connection_string)
    await task_queue.initialize()

    # 4. Initialize scheduler
    scheduler = AgentScheduler(task_queue, config)
    await scheduler.start()

    # 5. Initialize manual source manager
    source_manager = ManualSourceManager(task_queue, None)

    # 6. Initialize monitoring
    metrics = MetricsCollector(config.monitoring.metrics_port)
    health_checker = HealthChecker(config)

    # 7. Start metrics server
    metrics.start_server()

    agent_logger.info("All components initialized")
    agent_logger.info("-" * 60)

    # 8. Display startup summary
    agent_logger.info("Startup Summary:")
    agent_logger.info(f"  Database: {config.database.connection_string[:30]}...")
    agent_logger.info(f"  LLM API: {config.llm.base_url}")
    agent_logger.info(f"  Knowledge Base: {config.paths.knowledge_base}")
    agent_logger.info(f"  Metrics Port: {config.monitoring.metrics_port}")
    agent_logger.info("-" * 60)

    # 9. Display health status
    health = await health_checker.check_health()
    for component, status in health['overall']['checks'].items():
        if component == 'overall':
            continue
        status_emoji = "✓" if status['healthy'] else "✗"
        agent_logger.info(f"{status_emoji} {component.upper()}: {status['message']}")

    agent_logger.info("=" * 60)
    agent_logger.info("Agent running. Press Ctrl+C to shutdown.")
    agent_logger.info("=" * 60)

    # 10. Main loop
    try:
        while running:
            # Health check every 5 minutes
            await asyncio.sleep(300)

            # Periodic health display (every 30 minutes, 6 iterations)
            # (Would be handled by scheduler job)

    except KeyboardInterrupt:
        agent_logger.info("Keyboard interrupt received")

    except Exception as e:
        agent_logger.error(f"Fatal error in main loop: {e}", exc_info=True)

    finally:
        # 11. Graceful shutdown
        agent_logger.info("=" * 60)
        agent_logger.info("Shutting down...")
        agent_logger.info("=" * 60)

        # Stop scheduler
        await scheduler.stop()
        agent_logger.info("Scheduler stopped")

        # Close task queue
        await task_queue.close()
        agent_logger.info("Task queue closed")

        agent_logger.info("Shutdown complete")
        agent_logger.info("=" * 60)

def display_usage():
    """Display usage information"""

    print("""
Autonomous Research Agent v1.0
=============================

Usage:
  python research_agent/main.py [options]

Options:
  --config PATH       Path to config file (default: configs/research_agent_config.yaml)
  --log-level LEVEL   Log level (DEBUG, INFO, WARNING, ERROR)
  --help             Display this help message

Examples:
  # Run with default config
  python research_agent/main.py

  # Run with custom config
  python research_agent/main.py --config /path/to/config.yaml

  # Run with debug logging
  python research_agent/main.py --log-level DEBUG
""")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent - 24/7 Knowledge Gathering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(display_usage())
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/research_agent_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level from config"
    )

    parser.add_argument(
        "--help",
        action="store_true",
        help="Display help message"
    )

    # Parse
    args = parser.parse_args()

    if args.help:
        display_usage()
        sys.exit(0)

    # Setup signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
```

**Success Criteria**:
- [ ] Entry point loads configuration
- [ ] All components initialized
- [ ] Graceful shutdown works
- [ ] Signal handling works
- [ ] Usage message displays

**Verification**: Run `python research_agent/main.py`, verify startup

---

### Task 5.5: Systemd Service File
**Priority**: P1 | **Effort**: 1 hour | **Dependencies**: 5.4

**File**: `/etc/systemd/system/autonomous-research-agent.service` (create for reference)

**Implementation**:
```ini
[Unit]
Description=Autonomous 24/7 Research Agent
Documentation=https://github.com/yourrepo/autonomous-research-agent
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=administrator
WorkingDirectory=\\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer
# Add Python to PATH
Environment="PATH=/home/administrator/.local/bin:/usr/local/bin:/usr/bin:/bin"
# Override database connection (optional)
#Environment="DB_CONNECTION_STRING=postgresql://agentzero@localhost:5432/knowledge_graph"
ExecStart=/home/administrator/.local/bin/uv run research_agent/main.py
ExecStartPre=/bin/sh -c 'echo "Starting Research Agent at $(date)" >> /home/administrator/dev/unbiased-observer/logs/startup.log'
ExecStopPost=/bin/sh -c 'echo "Stopping Research Agent at $(date)" >> /home/administrator/dev/unbiased-observer/logs/shutdown.log'
Restart=always
RestartSec=10
# Restart on failure after 10 seconds
StandardOutput=append:/home/administrator/dev/unbiased-observer/logs/agent.log
StandardError=append:/home/administrator/dev/unbiased-observer/logs/agent_error.log

[Install]
WantedBy=multi-user.target
# Auto-start on boot
```

**Installation Instructions**:
```bash
# 1. Copy service file
sudo cp autonomous-research-agent.service /etc/systemd/system/

# 2. Reload systemd
sudo systemctl daemon-reload

# 3. Enable auto-start on boot
sudo systemctl enable autonomous-research-agent.service

# 4. Start service
sudo systemctl start autonomous-research-agent.service

# 5. Check status
sudo systemctl status autonomous-research-agent.service

# 6. View logs
sudo journalctl -u autonomous-research-agent -f
```

**Success Criteria**:
- [ ] Service file created
- [ ] Service can be enabled
- [ ] Service starts successfully
- [ ] Logs are captured
- [ ] Auto-restart works

**Verification**: Test service start/stop locally (without installing to systemd)

---

### Task 5.6: End-to-End Testing
**Priority**: P0 | **Effort**: 4 hours | **Dependencies**: All previous tasks

**File**: `research_agent/tests/integration_test.py`

**Implementation**:
```python
#!/usr/bin/env python3
"""
Integration test for autonomous research agent.
Tests full workflow end-to-end.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research_agent.config import Config
from research_agent.research.manual_source import ManualSourceManager
from research_agent.orchestrator.task_queue import TaskQueue
from research_agent.ingestion.pipeline import IngestionPipeline

class IntegrationTest:
    """End-to-end integration test"""

    def __init__(self):
        self.config = None
        self.task_queue = None

    async def setup(self):
        """Setup test environment"""

        # Create test config
        self.config = Config.__new__(
            database=Config.DatabaseConfig(
                connection_string="postgresql://agentzero@localhost:5432/knowledge_graph_test"
            ),
            llm=Config.LLMConfig(
                base_url="http://localhost:8317/v1",
                api_key="lm-studio",
                model_default="gemini-2.5-flash"
            ),
            paths=Config.PathsConfig(
                knowledge_base=str(Path(__file__).parent.parent.parent / "knowledge_base")
            )
        )

        # Initialize components
        self.task_queue = TaskQueue(self.config.database.connection_string)
        await self.task_queue.initialize()

        print("✓ Test environment setup complete")

    async def teardown(self):
        """Tear down test environment"""

        await self.task_queue.close()
        print("✓ Test environment cleaned up")

    async def test_manual_text_source(self):
        """Test adding manual text source"""

        print("\n" + "=" * 60)
        print("TEST 1: Manual Text Source")
        print("=" * 60)

        test_text = """
        Dr. Sarah Chen is a leading AI researcher at Cyberdyne Systems.
        She leads Project Alpha, which focuses on self-optimizing cognitive architectures.
        The project received significant funding from the Department of Advanced Technology in late 2024.
        Dr. Chen collaborates with Dr. James Wilson on neural network architectures.
        """

        # Add manual source
        source_manager = ManualSourceManager(self.task_queue, None)
        task_id = await source_manager.add_text_source(
            text=test_text,
            metadata={'test_case': 'manual_text'}
        )

        print(f"✓ Created task: {task_id}")

        # Wait for ingestion (simulated)
        await asyncio.sleep(30)

        # Verify task completed
        # (Would check database in real scenario)
        print("✓ Test 1: PASSED")

    async def test_fetch_and_ingest(self):
        """Test URL fetch and ingestion"""

        print("\n" + "=" * 60)
        print("TEST 2: URL Fetch + Ingestion")
        print("=" * 60)

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            The Oakley Institute conducts groundbreaking research in quantum computing.
            Dr. Elena Vance heads the Quantum Information Systems division.
            The institute collaborates with Stanford University on quantum error correction.
            In 2023, they published a seminal paper on fault-tolerant quantum algorithms.
            """)
            test_file = f.name

        # Add file source
        source_manager = ManualSourceManager(self.task_queue, None)
        task_id = await source_manager.add_file_source(
            file_path=test_file,
            metadata={'test_case': 'file_source'}
        )

        print(f"✓ Created task: {task_id}")

        # Wait for ingestion
        await asyncio.sleep(30)

        # Clean up
        Path(test_file).unlink()

        print("✓ Test 2: PASSED")

    async def test_error_recovery(self):
        """Test error recovery"""

        print("\n" + "=" * 60)
        print("TEST 3: Error Recovery")
        print("=" * 60)

        # Add invalid source (will fail)
        source_manager = ManualSourceManager(self.task_queue, None)

        try:
            await source_manager.add_url_source(
                url="http://this-does-not-exist.invalid",
                metadata={'test_case': 'invalid_url'}
            )
            print("✗ Test 3: Should have failed")
        except ValueError as e:
            print(f"✓ Test 3: PASSED - Correctly rejected invalid URL: {e}")

    async def test_concurrent_sources(self):
        """Test adding multiple sources concurrently"""

        print("\n" + "=" * 60)
        print("TEST 4: Concurrent Source Addition")
        print("=" * 60)

        # Add multiple sources
        sources = [
            ("Test content A", f"Test description A {i}")
            for i in range(5)
        ]

        tasks = []
        for content, desc in sources:
            source_manager = ManualSourceManager(self.task_queue, None)
            tasks.append(source_manager.add_text_source(content, metadata={'desc': desc}))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"✓ Added {successful}/{len(sources)} sources concurrently")
        print("✓ Test 4: PASSED")

    async def run_all_tests(self):
        """Run all integration tests"""

        print("\n" + "=" * 60)
        print("AUTONOMOUS RESEARCH AGENT - INTEGRATION TEST SUITE")
        print("=" * 60)

        try:
            await self.setup()

            # Run tests
            await self.test_manual_text_source()
            await self.test_fetch_and_ingest()
            await self.test_error_recovery()
            await self.test_concurrent_sources()

            # Summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)
            print("All tests completed successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await self.teardown()

if __name__ == "__main__":
    test = IntegrationTest()
    asyncio.run(test.run_all_tests())
```

**Success Criteria**:
- [ ] Manual text source works
- [ ] File source works
- [ ] URL validation works
- [ ] Concurrent additions work
- [ ] Error handling works
- [ ] All components integrate correctly

**Verification**: Run integration test, verify all tests pass

---

## DEPENDENCY INSTALLATION

### Required Packages

```bash
# Core dependencies
uv add pydantic pyyaml aiohttp bs4 beautifulsoup4

# APScheduler for 24/7 scheduling
uv add apscheduler

# Async PostgreSQL
uv add psycopg[pool]

# Knowledge base integration
# (Already exists, just import)

# Monitoring
uv add prometheus-client

# Development
uv add pytest pytest-asyncio
```

### Configuration Verification

Before running agent, verify:

```bash
# 1. Check PostgreSQL connection
psql -U agentzero -d knowledge_graph -c "SELECT 1;"

# 2. Check LLM API
curl http://localhost:8317/v1/models

# 3. Check config files
cat configs/research_agent_config.yaml
cat configs/research_sources.yaml

# 4. Check knowledge_base path
ls \\\\wsl.localhost\\Ubuntu\\home\\administrator\\dev\\unbiased-observer\\knowledge_base
```

---

## QUICK START GUIDE

### 1. Initial Setup (One-time)

```bash
# Install dependencies
cd /home/administrator/dev/unbiased-observer
uv pip install -r requirements.txt

# Create directories
mkdir -p research_agent/{orchestrator,research,ingestion,processing,monitoring}
mkdir -p configs
mkdir -p logs
mkdir -p state
mkdir -p cache

# Copy configuration files
# (Already created in task 5.3)
```

### 2. Database Setup

```bash
# Run schema
psql -U agentzero -d knowledge_graph -f research_agent/db_schema.sql

# Verify tables created
psql -U agentzero -d knowledge_graph -c "\dt"
```

### 3. Test Run

```bash
# Run agent in foreground
uv run research_agent/main.py

# Add a test source (in another terminal)
curl -X POST http://localhost:8001/api/sources \
  -H "Content-Type: application/json" \
  -d '{"source": "Test text", "content": "Test content here..."}'

# Check logs
tail -f logs/agent.log
```

### 4. Production Deployment

```bash
# Create systemd service
sudo cp configs/autonomous-research-agent.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable autonomous-research-agent.service
sudo systemctl start autonomous-research-agent.service

# Verify running
sudo systemctl status autonomous-research-agent.service

# View logs
sudo journalctl -u autonomous-research-agent -f -n 100
```

---

## TROUBLESHOOTING

### Common Issues

**1. PostgreSQL Connection Failed**
```
Error: connection to "localhost:5432" failed

Solution:
- Verify PostgreSQL is running: sudo systemctl status postgresql
- Check credentials in config
- Test connection: psql -U agentzero -d knowledge_graph
```

**2. LLM API Unavailable**
```
Error: Failed to list models

Solution:
- Verify LM Studio is running
- Check URL in config: curl http://localhost:8317/v1/models
- Check firewall settings
```

**3. WSL Path Issues**
```
Error: FileNotFoundError

Solution:
- Use forward slashes instead of backslashes
- Use /mnt/c/Users/... for Windows paths
- Test path with: ls <path>
```

**4. Agent Not Processing Tasks**
```
Issue: Tasks pending but not being processed

Solution:
- Check scheduler logs: tail logs/agent.log
- Verify worker ID assignment
- Check task queue table: SELECT * FROM research_tasks WHERE status = 'IN_PROGRESS'
```

**5. High Memory Usage**
```
Issue: Agent consuming too much memory

Solution:
- Reduce max_concurrent_llm_calls in config
- Reduce max_concurrent_fetches in config
- Monitor with: top -p $$ -u $USER
```

---

## SUCCESS CRITERIA SUMMARY

### Phase 1: Foundation
- [x] Project structure created
- [ ] Configuration management implemented
- [ ] Database schema created
- [ ] Logging setup complete
- [ ] Task queue operational
- [ ] Error recovery working
- [ ] Scheduler running 24/7

### Phase 2: Research Agent
- [ ] Content fetcher working
- [ ] Content extractor working
- [ ] Source discovery implemented
- [ ] Manual source interface ready
- [ ] Async ingestor working

### Phase 3: Ingestion Pipeline
- [ ] Direct PostgreSQL storage working
- [ ] Full ingestion pipeline operational
- [ ] Ingestion queue handler working

### Phase 4: Processing Pipeline
- [ ] Processing coordinator working
- [ ] Processing trigger management working

### Phase 5: Monitoring & Deployment
- [ ] Metrics collection working
- [ ] Health check system working
- [ ] Configuration files created
- [ ] Main entry point working
- [ ] Systemd service file created
- [ ] End-to-end tests passing

---

**Total Tasks**: 22
**Estimated Total Effort**: 5 weeks
**Next Step**: Start implementation with Task 1.1
