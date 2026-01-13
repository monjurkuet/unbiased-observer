#!/usr/bin/env python3
"""
Custom Research Source Examples

This script demonstrates how to implement and use custom research sources
with the Autonomous Research Agent.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any
from datetime import datetime, timedelta
import sys
import os

# Set PYTHONPATH for imports
sys.path.insert(0, "/home/administrator/dev/unbiased-observer")

from research_agent.config import load_config
from research_agent.monitoring import setup_logging
from research_agent.orchestrator import TaskQueue
from research_agent.research import ManualSourceManager


class CustomResearchSource:
    """
    Base class for custom research sources.

    This provides a template for implementing custom content discovery
    that integrates with the research agent's task queue.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    async def discover_content(self) -> List[Dict[str, Any]]:
        """
        Discover new research content.

        Returns a list of content items with the following structure:
        {
            'url': str,           # Direct link to content
            'title': str,         # Content title
            'content': str,       # Raw content (optional)
            'metadata': dict      # Additional metadata
        }
        """
        raise NotImplementedError("Subclasses must implement discover_content")

    async def validate_connection(self) -> bool:
        """Validate that the source is accessible."""
        return True


class GitHubResearchSource(CustomResearchSource):
    """
    Discover research content from GitHub repositories.

    Monitors GitHub issues and pull requests for research discussions
    and paper references.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("GitHub Research", config)
        self.token = config.get("token")
        self.repos = config.get("repos", [])
        self.days_back = config.get("days_back", 7)
        self.labels = config.get("labels", ["research", "paper"])

    async def discover_content(self) -> List[Dict[str, Any]]:
        """Discover research content from GitHub."""
        content_items = []

        for repo in self.repos:
            repo_items = await self._search_repo(repo)
            content_items.extend(repo_items)

        return content_items

    async def _search_repo(self, repo: str) -> List[Dict[str, Any]]:
        """Search a specific GitHub repository."""
        since = (datetime.now() - timedelta(days=self.days_back)).isoformat()

        # Search for issues with research labels
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "all", "since": since, "labels": ",".join(self.labels)}
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f"GitHub API error for {repo}: {response.status}")
                    return []

                issues = await response.json()

                content_items = []
                for issue in issues:
                    # Extract paper URLs from issue body
                    body = issue.get("body", "")
                    paper_urls = self._extract_paper_urls(body)

                    for paper_url in paper_urls:
                        content_items.append(
                            {
                                "url": paper_url,
                                "title": f"{issue['title']} - {repo}",
                                "content": body,
                                "metadata": {
                                    "source": "github",
                                    "repo": repo,
                                    "issue_number": issue["number"],
                                    "issue_url": issue["html_url"],
                                    "labels": [
                                        label["name"]
                                        for label in issue.get("labels", [])
                                    ],
                                    "author": issue["user"]["login"]
                                    if issue.get("user")
                                    else None,
                                },
                            }
                        )

                return content_items

    def _extract_paper_urls(self, text: str) -> List[str]:
        """Extract arXiv and PDF URLs from text."""
        import re

        urls = []

        # arXiv URLs
        arxiv_patterns = [
            r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)",
            r"arxiv\.org/(?:abs|pdf)/(\w+/\d+)",
        ]

        for pattern in arxiv_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                urls.append(f"https://arxiv.org/pdf/{match}.pdf")

        # Direct PDF URLs
        pdf_matches = re.findall(r"https?://[^\s]+\.pdf", text, re.IGNORECASE)
        urls.extend(pdf_matches)

        return list(set(urls))  # Remove duplicates

    async def validate_connection(self) -> bool:
        """Validate GitHub API access."""
        if not self.repos:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.token:
                    headers["Authorization"] = f"token {self.token}"

                # Test with first repo
                test_repo = self.repos[0]
                async with session.get(
                    f"https://api.github.com/repos/{test_repo}", headers=headers
                ) as response:
                    return response.status == 200
        except:
            return False


class SemanticScholarSource(CustomResearchSource):
    """
    Discover research papers from Semantic Scholar.

    Uses the Semantic Scholar API to find relevant research papers
    based on keyword searches.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Semantic Scholar", config)
        self.api_key = config.get("api_key")
        self.queries = config.get("queries", [])
        self.limit = config.get("limit", 10)
        self.min_citation_count = config.get("min_citation_count", 0)

    async def discover_content(self) -> List[Dict[str, Any]]:
        """Discover papers from Semantic Scholar."""
        content_items = []

        for query in self.queries:
            query_items = await self._search_query(query)
            content_items.extend(query_items)

        return content_items

    async def _search_query(self, query: str) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for a specific query."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": self.limit,
            "fields": "title,abstract,authors,year,citationCount,externalIds,venue",
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f"Semantic Scholar API error: {response.status}")
                    return []

                data = await response.json()

                content_items = []
                for paper in data.get("data", []):
                    # Skip papers with too few citations
                    if paper.get("citationCount", 0) < self.min_citation_count:
                        continue

                    # Get PDF URL from external IDs
                    pdf_url = self._get_pdf_url(paper.get("externalIds", {}))

                    if pdf_url:
                        content_items.append(
                            {
                                "url": pdf_url,
                                "title": paper.get("title", ""),
                                "content": paper.get("abstract", ""),
                                "metadata": {
                                    "source": "semantic_scholar",
                                    "paper_id": paper.get("paperId"),
                                    "authors": [
                                        author.get("name", "")
                                        for author in paper.get("authors", [])
                                    ],
                                    "year": paper.get("year"),
                                    "citation_count": paper.get("citationCount", 0),
                                    "venue": paper.get("venue"),
                                    "query": query,
                                },
                            }
                        )

                return content_items

    def _get_pdf_url(self, external_ids: Dict[str, Any]) -> str:
        """Extract PDF URL from external IDs."""
        # Try arXiv first
        if "ArXiv" in external_ids:
            return f"https://arxiv.org/pdf/{external_ids['ArXiv']}.pdf"

        # Try DOI
        if "DOI" in external_ids:
            doi = external_ids["DOI"]
            return f"https://doi.org/{doi}"

        return None

    async def validate_connection(self) -> bool:
        """Validate Semantic Scholar API access."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers["x-api-key"] = self.api_key

                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "test", "limit": 1},
                    headers=headers,
                ) as response:
                    return response.status == 200
        except:
            return False


class RSSResearchSource(CustomResearchSource):
    """
    Discover research content from RSS feeds.

    Monitors RSS feeds from research blogs, journals, and news sites
    for new research content.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("RSS Research", config)
        self.feeds = config.get("feeds", [])
        self.days_back = config.get("days_back", 7)
        self.keywords = config.get("keywords", [])

    async def discover_content(self) -> List[Dict[str, Any]]:
        """Discover content from RSS feeds."""
        content_items = []

        for feed_url in self.feeds:
            feed_items = await self._process_feed(feed_url)
            content_items.extend(feed_items)

        return content_items

    async def _process_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Process a single RSS feed."""
        try:
            import feedparser

            # Parse RSS feed
            feed = feedparser.parse(feed_url)

            content_items = []
            cutoff_date = datetime.now() - timedelta(days=self.days_back)

            for entry in feed.entries:
                # Check date
                published = self._parse_date(entry)
                if published and published < cutoff_date:
                    continue

                # Check keywords if specified
                title_content = f"{entry.title} {getattr(entry, 'summary', '')}"
                if self.keywords and not self._contains_keywords(
                    title_content, self.keywords
                ):
                    continue

                # Extract content URL
                content_url = self._extract_content_url(entry)

                if content_url:
                    content_items.append(
                        {
                            "url": content_url,
                            "title": entry.title,
                            "content": getattr(entry, "summary", ""),
                            "metadata": {
                                "source": "rss",
                                "feed_url": feed_url,
                                "published": published.isoformat()
                                if published
                                else None,
                                "feed_title": feed.get("title", ""),
                                "entry_link": entry.link,
                            },
                        }
                    )

            return content_items

        except Exception as e:
            print(f"Error processing RSS feed {feed_url}: {e}")
            return []

    def _parse_date(self, entry) -> datetime:
        """Parse publication date from RSS entry."""
        date_fields = ["published_parsed", "updated_parsed", "created_parsed"]

        for field in date_fields:
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    return datetime(*getattr(entry, field)[:6])
                except:
                    continue

        return None

    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def _extract_content_url(self, entry) -> str:
        """Extract the best content URL from RSS entry."""
        # Try enclosures (for podcasts, videos)
        if hasattr(entry, "enclosures") and entry.enclosures:
            for enclosure in entry.enclosures:
                if enclosure.get("type", "").startswith("application/pdf"):
                    return enclosure.get("href")

        # Try links
        if hasattr(entry, "links"):
            for link in entry.links:
                href = link.get("href", "")
                if href.endswith(".pdf") or "pdf" in href:
                    return href

        # Fallback to main link
        return getattr(entry, "link", None)

    async def validate_connection(self) -> bool:
        """Validate RSS feed accessibility."""
        if not self.feeds:
            return False

        try:
            import feedparser

            # Test with first feed
            feed = feedparser.parse(self.feeds[0])
            return hasattr(feed, "entries") and len(feed.entries) > 0
        except:
            return False


async def example_github_source():
    """Example: Using GitHub research source."""
    print("=== GitHub Research Source Example ===")

    config = {
        "repos": ["papers-we-love/papers-we-love", "karpathy/research-papers"],
        "token": os.getenv("GITHUB_TOKEN"),  # Optional
        "days_back": 14,
        "labels": ["research", "paper", "ml", "ai"],
    }

    source = GitHubResearchSource(config)

    # Validate connection
    is_valid = await source.validate_connection()
    print(f"GitHub connection valid: {is_valid}")

    if is_valid:
        # Discover content
        content = await source.discover_content()
        print(f"Found {len(content)} research items from GitHub")

        for item in content[:3]:  # Show first 3
            print(f"  - {item['title']}")
            print(f"    URL: {item['url']}")
            print(f"    Repo: {item['metadata']['repo']}")
            print()

    print("GitHub source example complete!\n")


async def example_semantic_scholar_source():
    """Example: Using Semantic Scholar source."""
    print("=== Semantic Scholar Source Example ===")

    config = {
        "api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),  # Optional
        "queries": ["transformer architecture", "attention mechanism"],
        "limit": 5,
        "min_citation_count": 10,
    }

    source = SemanticScholarSource(config)

    # Validate connection
    is_valid = await source.validate_connection()
    print(f"Semantic Scholar connection valid: {is_valid}")

    if is_valid:
        # Discover content
        content = await source.discover_content()
        print(f"Found {len(content)} papers from Semantic Scholar")

        for item in content[:3]:  # Show first 3
            print(f"  - {item['title']}")
            print(f"    Citations: {item['metadata']['citation_count']}")
            print(f"    Year: {item['metadata']['year']}")
            print()

    print("Semantic Scholar source example complete!\n")


async def example_rss_source():
    """Example: Using RSS research source."""
    print("=== RSS Research Source Example ===")

    config = {
        "feeds": [
            "https://www.technologyreview.com/topnews.rss",
            "https://www.sciencenews.org/feed",
        ],
        "days_back": 7,
        "keywords": ["AI", "machine learning", "research"],
    }

    source = RSSResearchSource(config)

    # Validate connection
    is_valid = await source.validate_connection()
    print(f"RSS connection valid: {is_valid}")

    if is_valid:
        # Discover content
        content = await source.discover_content()
        print(f"Found {len(content)} research items from RSS feeds")

        for item in content[:3]:  # Show first 3
            print(f"  - {item['title']}")
            print(f"    Feed: {item['metadata']['feed_title']}")
            print(f"    Published: {item['metadata']['published']}")
            print()

    print("RSS source example complete!\n")


async def example_integrate_custom_source():
    """Example: Integrating custom source with research agent."""
    print("=== Integrating Custom Source with Agent ===")

    # Load agent configuration
    agent_config = load_config()
    logger, _, _, _ = setup_logging(agent_config)

    # Initialize task queue
    queue = TaskQueue(agent_config, logger)
    await queue.initialize()

    # Create custom source
    github_config = {
        "repos": ["papers-we-love/papers-we-love"],
        "days_back": 7,
        "labels": ["research"],
    }

    source = GitHubResearchSource(github_config)

    # Discover content
    content_items = await source.discover_content()
    print(f"Discovered {len(content_items)} items")

    # Add to agent's processing queue
    manager = ManualSourceManager(queue)

    added_count = 0
    for item in content_items:
        try:
            if item.get("url"):
                task_id = await manager.add_url_source(
                    url=item["url"],
                    metadata={
                        **item.get("metadata", {}),
                        "custom_source": True,
                        "discovered_at": datetime.now().isoformat(),
                    },
                )
                added_count += 1
                print(f"Added to queue: {item['title']} (Task ID: {task_id})")
        except Exception as e:
            print(f"Error adding item: {e}")

    print(f"Successfully added {added_count} items to processing queue")

    await queue.close()
    print("Custom source integration complete!\n")


async def main():
    """Run all custom source examples."""
    print("Autonomous Research Agent - Custom Source Examples")
    print("=" * 55)
    print()

    try:
        await example_github_source()
        await example_semantic_scholar_source()
        await example_rss_source()
        await example_integrate_custom_source()

        print("All custom source examples completed successfully!")
        print("\nTo create your own custom source:")
        print("1. Extend the CustomResearchSource base class")
        print("2. Implement discover_content() and validate_connection()")
        print("3. Add your source to research_sources.yaml")
        print("4. Integrate with the ManualSourceManager")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Check if running in correct environment
    if not os.path.exists("/home/administrator/dev/unbiased-observer"):
        print("Error: This script must be run from the correct environment.")
        print("Expected path: /home/administrator/dev/unbiased-observer")
        sys.exit(1)

    # Run examples
    asyncio.run(main())
