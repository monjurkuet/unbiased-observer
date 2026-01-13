import aiohttp
import asyncio
import logging
from typing import Dict, List
from research_agent.orchestrator.error_recovery import with_retry

logger = logging.getLogger("research_agent.research")


class ContentFetcher:
    """Async content fetcher with rate limiting"""

    def __init__(self, config):
        self.config = config.research
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_fetches)
        self.last_request_time = 0.0

    async def fetch_url(self, url: str) -> str:
        """Fetch content from URL with rate limiting"""

        await self.semaphore.acquire()
        try:
            await self._rate_limit_wait()

            logger.info(f"Fetching: {url}")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Fetched {len(content)} chars from {url}")
                        return content
                    else:
                        raise FetchError(f"HTTP {response.status}: {url}")

        finally:
            self.semaphore.release()

    async def fetch_file(self, file_path: str) -> str:
        """Fetch content from local file (WSL path)"""

        logger.info(f"Reading file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"Read {len(content)} chars from {file_path}")
            return content

        except Exception as e:
            raise FetchError(f"Failed to read file {file_path}: {e}")

    async def _rate_limit_wait(self):
        """Enforce rate limiting"""

        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        min_interval = 1.0 / self.config.rate_limit

        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self.last_request_time = asyncio.get_event_loop().time()

    @with_retry(max_retries=3)
    async def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        """Fetch multiple URLs concurrently"""

        logger.info(f"Fetching batch of {len(urls)} URLs")

        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {url}: {result}")
            else:
                successful[url] = result

        logger.info(f"Successfully fetched {len(successful)}/{len(urls)} URLs")
        return successful


class FetchError(Exception):
    """Content fetch error"""

    pass
