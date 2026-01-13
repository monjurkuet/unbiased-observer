import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger("research_agent.processing")


class ProcessingTrigger:
    """Determine when to run processing pipeline"""

    def __init__(self, config):
        self.config = config.processing
        self.db_conn_str = config.database.connection_string

    async def should_trigger(self) -> bool:
        """Check if processing should trigger"""

        entity_count_ok = await self._check_entity_count()

        if not entity_count_ok:
            logger.debug("Entity count threshold not met")
            return False

        time_ok = await self._check_time_interval()

        if not time_ok:
            logger.debug("Time interval not met")
            return False

        return True

    async def _check_entity_count(self) -> bool:
        """Check if entity count meets minimum"""

        from psycopg import AsyncConnection

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM nodes")
                row = await cur.fetchone()
                if row is None:
                    logger.warning("Failed to fetch entity count")
                    return False
                entity_count = row[0]

        min_entities = self.config.min_entities_to_process

        result = entity_count >= min_entities

        logger.info(f"Entity count check: {entity_count} >= {min_entities} = {result}")
        return result

    async def _check_time_interval(self) -> bool:
        """Check if minimum time has passed since last processing"""

        from psycopg import AsyncConnection

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT value FROM agent_state
                    WHERE key = 'last_processing_time'
                    """
                )
                row = await cur.fetchone()

                if not row:
                    logger.info("No previous processing time found")
                    return True

                last_time_str = row[0]

                if isinstance(last_time_str, str):
                    last_time = datetime.fromisoformat(json.loads(last_time_str))
                else:
                    last_time = last_time_str

                min_interval = timedelta(
                    hours=self.config.min_time_between_processing_hours
                )

                current_time = datetime.now()
                time_since = current_time - last_time

                result = time_since >= min_interval

                logger.info(
                    f"Time interval check: "
                    f"{time_since.total_seconds()}s >= {min_interval.total_seconds()}s = {result}"
                )

                return result

    async def record_processing_time(self):
        """Record processing time in agent_state"""

        current_time = datetime.now().isoformat()

        from psycopg import AsyncConnection

        async with await AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO agent_state (key, value)
                    VALUES ('last_processing_time', %s)
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                    """,
                    (current_time,),
                )

        logger.info(f"Recorded processing time: {current_time}")
