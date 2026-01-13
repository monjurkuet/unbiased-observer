"""arXiv integration for research monitoring."""

import datetime
import logging
from typing import List, Dict, Optional

import arxiv

logger = logging.getLogger("research_agent.research")


class ArxivIntegrator:
    """arXiv API wrapper for research integration."""

    def __init__(self):
        self.client = arxiv.Client()

    def search_by_keywords(
        self, keywords: List[str], max_results: int = 10, days_back: int = 7
    ) -> List[Dict]:
        """Search arXiv for recent papers by keywords."""
        query = " AND ".join(f"all:{kw}" for kw in keywords)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=days_back
        )

        for result in self.client.results(search):
            if result.published < cutoff_date:
                continue

            results.append(
                {
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "published": result.published.isoformat(),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "categories": result.categories,
                }
            )

        logger.info(f"Found {len(results)} arXiv papers for keywords: {keywords}")
        return results

    def search_by_category(
        self, category: str, max_results: int = 10, days_back: int = 7
    ) -> List[Dict]:
        """Search arXiv by category (e.g., cs.AI, stat.ML)."""
        query = f"cat:{category}"
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = []
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=days_back
        )

        for result in self.client.results(search):
            if result.published < cutoff_date:
                continue

            results.append(
                {
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "published": result.published.isoformat(),
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "categories": result.categories,
                }
            )

        logger.info(f"Found {len(results)} arXiv papers in category: {category}")
        return results

    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get detailed information about a specific paper."""
        search = arxiv.Search(id_list=[paper_id])

        try:
            result = next(self.client.results(search))
            return {
                "title": result.title,
                "authors": [str(author) for author in result.authors],
                "published": result.published.isoformat(),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "categories": result.categories,
            }
        except StopIteration:
            logger.warning(f"Paper not found: {paper_id}")
            return None


class ArxivSourceManager:
    """Manages arXiv as an automated research source."""

    def __init__(self, task_queue, config):
        self.task_queue = task_queue
        self.config = config
        self.integrator = ArxivIntegrator()
        self.search_configs = self._load_search_configs()

    def _load_search_configs(self) -> List[Dict]:
        """Load arXiv search configurations."""
        # Default configurations - can be extended from config file
        return [
            {
                "name": "AI Research",
                "keywords": ["artificial intelligence", "machine learning", "deep learning"],
                "category": "cs.AI",
                "max_results": 5,
                "days_back": 7,
                "active": True,
            },
            {
                "name": "NLP Research",
                "keywords": ["natural language processing", "transformers", "LLM"],
                "category": "cs.CL",
                "max_results": 5,
                "days_back": 7,
                "active": True,
            },
            {
                "name": "Computer Vision",
                "keywords": ["computer vision", "image recognition", "neural networks"],
                "category": "cs.CV",
                "max_results": 3,
                "days_back": 7,
                "active": True,
            },
        ]

    async def discover_new_papers(self) -> List[Dict]:
        """Discover new papers from configured arXiv searches."""
        logger.info("Discovering new papers from arXiv...")

        discovered = []

        for search_config in self.search_configs:
            if not search_config.get("active", True):
                continue

            logger.info(f"Searching arXiv for: {search_config['name']}")

            try:
                # Search by keywords
                if "keywords" in search_config:
                    papers = self.integrator.search_by_keywords(
                        keywords=search_config["keywords"],
                        max_results=search_config.get("max_results", 5),
                        days_back=search_config.get("days_back", 7),
                    )
                    discovered.extend(papers)

                # Search by category
                elif "category" in search_config:
                    papers = self.integrator.search_by_category(
                        category=search_config["category"],
                        max_results=search_config.get("max_results", 5),
                        days_back=search_config.get("days_back", 7),
                    )
                    discovered.extend(papers)

            except Exception as e:
                logger.error(f"Error searching arXiv for {search_config['name']}: {e}")
                continue

        # Remove duplicates based on entry_id
        seen_ids = set()
        unique_discovered = []
        for paper in discovered:
            if paper["entry_id"] not in seen_ids:
                seen_ids.add(paper["entry_id"])
                unique_discovered.append(paper)

        logger.info(f"Discovered {len(unique_discovered)} unique arXiv papers")
        return unique_discovered

    async def add_paper_to_queue(self, paper: Dict) -> str:
        """Add an arXiv paper to the processing queue."""
        logger.info(f"Adding arXiv paper to queue: {paper['title'][:50]}...")

        # Create FETCH task for the PDF
        task_id = await self.task_queue.add_task(
            task_type="FETCH",
            source=paper["pdf_url"],
            metadata={
                "source_id": paper["entry_id"],
                "source_type": "arxiv_pdf",
                "title": paper["title"],
                "authors": paper["authors"],
                "published": paper["published"],
                "categories": paper["categories"],
                "added_by": "arxiv_monitor",
                "content_type": "pdf",
            }
        )

        logger.info(f"Created FETCH task {task_id} for arXiv paper")
        return task_id

    async def run_monitoring_cycle(self) -> int:
        """Run a complete arXiv monitoring cycle."""
        logger.info("Starting arXiv monitoring cycle...")

        # Discover new papers
        papers = await self.discover_new_papers()

        # Add each paper to processing queue
        added_count = 0
        for paper in papers:
            try:
                await self.add_paper_to_queue(paper)
                added_count += 1
            except Exception as e:
                logger.error(f"Failed to add paper {paper.get('entry_id')}: {e}")
                continue

        logger.info(f"arXiv monitoring cycle complete: {added_count} papers added to queue")
        return added_count
