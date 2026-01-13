#!/usr/bin/env python3
"""
Main entry point for the 24/7 Autonomous Research Agent.
"""

import asyncio
import signal
import sys
from pathlib import Path

from research_agent.config import load_config
from research_agent.monitoring import setup_logging
from research_agent.orchestrator.scheduler import AgentScheduler
from research_agent.orchestrator.task_queue import TaskQueue


async def main() -> int:
    """
    Main entry point for the research agent.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    config = load_config()

    agent_logger, ingestion_logger, processing_logger, orchestrator_logger = (
        setup_logging(
            config,
            debug=False,
        )
    )

    agent_logger.info("=" * 60)
    agent_logger.info("24/7 Autonomous Research Agent Starting")
    agent_logger.info("=" * 60)
    agent_logger.info(f"Database: {config.database.connection_string[:40]}...")
    agent_logger.info(f"LLM API: {config.llm.base_url}")
    agent_logger.info(f"Knowledge Base: {config.paths.knowledge_base[:40]}...")

    task_queue = TaskQueue(config.database.connection_string)
    await task_queue.initialize()
    scheduler = AgentScheduler(task_queue, config)

    shutdown_event = asyncio.Event()

    def signal_handler(signum: int, frame):
        agent_logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await scheduler.start()

    agent_logger.info("Research agent started successfully")
    agent_logger.info("Press Ctrl+C to stop")

    try:
        await shutdown_event.wait()
    except Exception as e:
        agent_logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    agent_logger.info("Shutting down gracefully...")
    await scheduler.stop()
    agent_logger.info("Research agent stopped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
