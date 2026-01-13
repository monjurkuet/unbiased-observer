"""
Task Queue - Persistent task queue for 24/7 operation
"""

import psycopg
from psycopg import AsyncConnection
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger("research_agent.orchestrator")


@dataclass
class ResearchTask:
    id: str
    type: str
    source: Optional[str]
    metadata: dict
    status: str
    worker_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    max_retries: int
    error_message: Optional[str]


class TaskQueue:
    """Persistent task queue with PostgreSQL backend"""

    def __init__(self, db_conn_str: str):
        self.db_conn_str = db_conn_str

    async def initialize(self):
        """Initialize task queue"""
        logger.info("Task queue initialized")

    async def add_task(
        self, task_type: str, source: str = "", metadata: dict = None
    ) -> str:
        """Add a new task to the queue"""

        import uuid

        task_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO research_tasks
                    (id, type, source, metadata, status, created_at, retry_count, max_retries)
                    VALUES (%s, %s, %s, %s, 'PENDING', NOW(), 0, 3)
                    """,
                    (task_id, task_type, source, json.dumps(metadata)),
                )

        logger.info(f"Added task {task_id} ({task_type})")
        return task_id

    async def get_next_task(
        self, worker_id: str, task_types: List[str] = None
    ) -> Optional[ResearchTask]:
        """Get next available task for processing"""

        if task_types is None:
            task_types = ["FETCH", "INGEST", "PROCESS"]

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, type, source, metadata, status, worker_id,
                           created_at, started_at, completed_at, retry_count, max_retries, error_message
                    FROM research_tasks
                    WHERE status = 'PENDING'
                      AND type = ANY(%s)
                      AND retry_count < max_retries
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    (task_types,),
                )

                row = await cur.fetchone()
                if not row:
                    return None

                # Update task as in progress
                task_id = row[0]
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET status = 'IN_PROGRESS', worker_id = %s, started_at = NOW()
                    WHERE id = %s
                    """,
                    (worker_id, task_id),
                )

        # Parse metadata
        import json

        metadata = json.loads(row[3]) if row[3] else {}

        return ResearchTask(
            id=row[0],
            type=row[1],
            source=row[2],
            metadata=metadata,
            status=row[4],
            worker_id=row[5],
            created_at=row[6],
            started_at=row[7],
            completed_at=row[8],
            retry_count=row[9],
            max_retries=row[10],
            error_message=row[11],
        )

    async def update_task_status(
        self, task_id: str, status: str, error_message: str = None
    ) -> None:
        """Update task status"""

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                if status == "COMPLETED":
                    await cur.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s, completed_at = NOW(), error_message = NULL
                        WHERE id = %s
                        """,
                        (status, task_id),
                    )
                elif status == "FAILED":
                    await cur.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s, error_message = %s
                        WHERE id = %s
                        """,
                        (status, error_message, task_id),
                    )
                else:
                    await cur.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s, error_message = %s
                        WHERE id = %s
                        """,
                        (status, error_message, task_id),
                    )

    async def increment_retry(self, task_id: str) -> None:
        """Increment retry count for failed task"""

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET retry_count = retry_count + 1,
                        status = CASE WHEN retry_count + 1 >= max_retries THEN 'FAILED' ELSE 'PENDING' END
                    WHERE id = %s
                    """,
                    (task_id,),
                )

    async def get_failed_tasks(self) -> List[ResearchTask]:
        """Get tasks that failed and can be retried"""

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, type, source, metadata, status, worker_id,
                           created_at, started_at, completed_at, retry_count, max_retries, error_message
                    FROM research_tasks
                    WHERE status = 'FAILED'
                      AND retry_count < max_retries
                      AND created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at ASC
                    """
                )

                tasks = []
                async for row in cur:
                    import json

                    metadata = json.loads(row[3]) if row[3] else {}

                    tasks.append(
                        ResearchTask(
                            id=row[0],
                            type=row[1],
                            source=row[2],
                            metadata=metadata,
                            status=row[4],
                            worker_id=row[5],
                            created_at=row[6],
                            started_at=row[7],
                            completed_at=row[8],
                            retry_count=row[9],
                            max_retries=row[10],
                            error_message=row[11],
                        )
                    )

                return tasks

    async def get_pending_count(self) -> int:
        """Get count of pending tasks"""

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM research_tasks WHERE status = 'PENDING'"
                )
                row = await cur.fetchone()
                return row[0] if row else 0

    async def get_last_failed_tasks(self, limit: int = 10) -> List[ResearchTask]:
        """Get recent failed tasks for debugging"""

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, type, source, metadata, status, worker_id,
                           created_at, started_at, completed_at, retry_count, max_retries, error_message
                    FROM research_tasks
                    WHERE status = 'FAILED'
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )

                tasks = []
                async for row in cur:
                    import json

                    metadata = json.loads(row[3]) if row[3] else {}

                    tasks.append(
                        ResearchTask(
                            id=row[0],
                            type=row[1],
                            source=row[2],
                            metadata=metadata,
                            status=row[4],
                            worker_id=row[5],
                            created_at=row[6],
                            started_at=row[7],
                            completed_at=row[8],
                            retry_count=row[9],
                            max_retries=row[10],
                            error_message=row[11],
                        )
                    )

                return tasks
