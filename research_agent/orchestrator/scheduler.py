"""
Scheduler - 24/7 scheduler for autonomous research agent
"""

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from psycopg import AsyncConnection
from .task_queue import TaskQueue
import logging
from typing import Dict, Any

logger = logging.getLogger("research_agent.orchestrator")


class AgentScheduler:
    """24/7 scheduler for autonomous research agent"""

    def __init__(self, task_queue: "TaskQueue", config):
        self.scheduler = AsyncIOScheduler()
        self.task_queue = task_queue
        self.config = config
        self.worker_id = f"worker-{asyncio.get_event_loop().time()}"

        # Job references
        self.jobs: Dict[str, Any] = {}

    async def start(self):
        """Start scheduler with all jobs"""
        # Job 1: Process task queue (every 10 seconds)
        self.jobs["process_queue"] = self.scheduler.add_job(
            self._process_queue,
            trigger=IntervalTrigger(seconds=10),
            id="process_queue",
            name="Process Task Queue",
            max_instances=1,
        )

        # Job 2: Processing pipeline (every 60 seconds)
        self.jobs["processing_pipeline"] = self.scheduler.add_job(
            self._process_ingestion_queue,
            trigger=IntervalTrigger(seconds=60),
            id="processing_pipeline",
            name="Processing Pipeline",
            max_instances=1,
        )

        # Job 3: Monitoring (every 5 minutes)
        self.jobs["monitoring"] = self.scheduler.add_job(
            self._monitoring,
            trigger=IntervalTrigger(seconds=300),
            id="monitoring",
            name="Health Monitoring",
            max_instances=1,
        )

        # Job 4: Retry failed tasks
        self.jobs["retry_failed"] = self.scheduler.add_job(
            self._retry_failed_tasks,
            trigger=IntervalTrigger(seconds=300),
            id="retry_failed",
            name="Retry Failed Tasks",
            max_instances=1,
        )

        # Job 5: arXiv monitoring (every 2 hours)
        self.jobs["arxiv_monitoring"] = self.scheduler.add_job(
            self._arxiv_monitoring,
            trigger=IntervalTrigger(seconds=7200),  # 2 hours
            id="arxiv_monitoring",
            name="arXiv Paper Monitoring",
            max_instances=1,
        )

        # Start scheduler (every 5 minutes)
        self.jobs["retry_failed"] = self.scheduler.add_job(
            self._retry_failed_tasks,
            trigger=IntervalTrigger(seconds=300),
            id="retry_failed",
            name="Retry Failed Tasks",
            max_instances=1,
        )

        # Start scheduler
        self.scheduler.start()

        logger.info("Scheduler started with 5 jobs:")
        for job_id, job in self.jobs.items():
            logger.info(f"  - {job_id}: {job.name}")

    async def stop(self):
        """Stop scheduler gracefully"""
        logger.info("Stopping scheduler...")

        # Remove all jobs
        for job_id in list(self.jobs.keys()):
            self.scheduler.remove_job(job_id)

        # Shutdown scheduler
        self.scheduler.shutdown(wait=True)

        logger.info("Scheduler stopped")

    async def _process_queue(self):
        """Process research tasks from queue"""
        try:
            # Get next task
            task = await self.task_queue.get_next_task(
                self.worker_id, task_types=["FETCH", "INGEST"]
            )

            if not task:
                logger.debug("No pending tasks in queue")
                return

            logger.info(f"Processing task: {task.type} - {task.source}")

            # Route to appropriate handler
            if task.type == "FETCH":
                await self._handle_fetch_task(task)
            elif task.type == "INGEST":
                await self._handle_ingest_task(task)

            # Mark task as completed
            await self.task_queue.update_task_status(task.id, "COMPLETED")

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)
            # Mark as failed (will be retried)
            await self.task_queue.increment_retry(task.id)

    async def _process_ingestion_queue(self):
        """Process ingested content through pipeline"""
        try:
            # Get next task
            task = await self.task_queue.get_next_task(
                self.worker_id, task_types=["PROCESS"]
            )

            if not task:
                return

            # Processing pipeline logic (to be implemented in Phase 4)
            await self._handle_process_task(task)

            await self.task_queue.update_task_status(task.id, "COMPLETED")

        except Exception as e:
            logger.error(f"Error in processing: {e}", exc_info=True)
            await self.task_queue.increment_retry(task.id)

    async def _retry_failed_tasks(self):
        """Retry failed tasks that haven't exceeded retries"""
        try:
            failed_tasks = await self.task_queue.get_failed_tasks()

            for task in failed_tasks:
                logger.info(f"Retrying task: {task.id}")

                # Reset to PENDING
                await self.task_queue.update_task_status(
                    task.id, "PENDING", error_message=None
                )

        except Exception as e:
            logger.error(f"Error retrying failed tasks: {e}", exc_info=True)

    async def _monitoring(self):
        """Health checks and metrics collection"""
        try:
            logger.debug("Running health checks...")

            # Check task queue
            pending_count = await self.task_queue.get_pending_count()
            logger.info(f"Pending tasks: {pending_count}")

            # Check database connectivity
            # (Implement in monitoring phase)

            # Check LLM API
            # (Implement in monitoring phase)

        except Exception as e:
            logger.error(f"Error in monitoring: {e}", exc_info=True)

    async def _handle_fetch_task(self, task):
        """Handle content fetch task"""
        # Placeholder - will be implemented in Phase 2
        logger.info(f"Fetching content from: {task.source}")
        pass

    async def _handle_ingest_task(self, task):
        """Handle ingestion task"""

        logger.info(f"Handling INGEST task: {task.id}")

        try:
            source = task.source
            metadata = task.metadata

            content = None

            if metadata.get("source_type") == "text":
                content = metadata.get("text_content")
                logger.info("Using direct text content from task")

            else:
                from research_agent.research.content_fetcher import ContentFetcher

                fetcher = ContentFetcher(self.config)

                if metadata.get("source_type") == "file":
                    content = await fetcher.fetch_file(source)
                elif metadata.get("source_type") == "url":
                    content = await fetcher.fetch_url(source)

            if not content:
                await self.task_queue.update_task_status(
                    task.id, "FAILED", error_message="No content to ingest"
                )
                return

            from research_agent.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline(self.config)
            await pipeline.initialize()

            result = await pipeline.ingest_content(
                content, {"source": source, **metadata}
            )

            if result["status"] == "completed":
                await self.task_queue.update_task_status(task.id, "COMPLETED")
                await self._log_ingestion_result(task.id, result)

            else:
                await self.task_queue.update_task_status(
                    task.id,
                    "FAILED",
                    error_message=result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"Error in ingest task: {e}", exc_info=True)
            await self.task_queue.increment_retry(task.id)

    async def _log_ingestion_result(self, task_id: str, result: Dict):
        """Log ingestion result to database"""

        async with await AsyncConnection.connect(self.task_queue.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO ingestion_logs
                    (task_id, status, entities_stored, edges_stored, events_stored, duration_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        task_id,
                        result["status"],
                        result.get("entities_stored", 0),
                        result.get("edges_stored", 0),
                        result.get("events_stored", 0),
                        result.get("duration_seconds", 0),
                    ),
                )

    async def _handle_process_task(self, task):
        """Handle processing task"""

        logger.info(f"Handling PROCESS task: {task.id}")

        try:
            from research_agent.processing.trigger import ProcessingTrigger
            from research_agent.processing.coordinator import ProcessingCoordinator

            trigger = ProcessingTrigger(self.config)

            if not await trigger.should_trigger():
                logger.info("Processing conditions not met, skipping")
                await self.task_queue.update_task_status(task.id, "COMPLETED")
                return

            coordinator = ProcessingCoordinator(self.config)
            await coordinator.initialize()

            result = await coordinator.run_processing_pipeline()

            if result["status"] == "completed":
                await self.task_queue.update_task_status(task.id, "COMPLETED")

                await trigger.record_processing_time()

                await self._log_processing_result(task.id, result)

            else:
                await self.task_queue.update_task_status(
                    task.id,
                    "FAILED",
                    error_message=result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"Error in process task: {e}", exc_info=True)
            await self.task_queue.increment_retry(task.id)

    async def _log_processing_result(self, task_id: str, result: Dict):
        """Log processing result to database"""

        async with await AsyncConnection.connect(self.task_queue.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO processings_logs
                    (task_id, status, nodes_processed, communities_created, duration_seconds)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        task_id,
                        result["status"],
                        result.get("nodes_processed", 0),
                        result.get("communities_created", 0),
                        result.get("duration_seconds", 0),
                    ),
                )

    async def _arxiv_monitoring(self):
        """Monitor arXiv for new research papers"""
        try:
            logger.info("Running arXiv monitoring...")

            from research_agent.research import ArxivSourceManager
            arxiv_manager = ArxivSourceManager(self.task_queue, self.config)

            papers_added = await arxiv_manager.run_monitoring_cycle()

            logger.info(f"arXiv monitoring complete: {papers_added} papers added to queue")

        except Exception as e:
            logger.error(f"Error in arXiv monitoring: {e}", exc_info=True)
