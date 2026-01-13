# Extending Research Sources

Guide to adding custom research sources and content discovery methods.

---

## Table of Contents

- [Overview](#overview)
- [Built-in Source Types](#built-in-source-types)
- [Adding Custom Sources](#adding-custom-sources)
- [Implementing Source Connectors](#implementing-source-connectors)
- [Content Format Support](#content-format-support)
- [Rate Limiting and Throttling](#rate-limiting-and-throttling)
- [Testing Custom Sources](#testing-custom-sources)
- [Best Practices](#best-practices)

---

## Overview

The Autonomous Research Agent supports multiple research content sources out of the box, but can be extended with custom sources for specialized research domains, proprietary databases, or unique content formats.

### Source Architecture

```
Source Discovery
├── Built-in Sources
│   ├── arXiv API
│   ├── URL Fetching
│   └── File System
└── Custom Sources
    ├── RSS Feeds
    ├── APIs
    ├── Databases
    └── Web Scraping
```

### Extension Points

- **Source Types**: Add new source categories
- **Content Fetchers**: Custom retrieval methods
- **Content Extractors**: Specialized format parsing
- **Source Managers**: Coordination and scheduling

---

## Built-in Source Types

### arXiv Sources

Automated monitoring of arXiv preprint server:

```yaml
sources:
  - type: "arxiv"
    name: "Computer Vision Research"
    config:
      category: "cs.CV"  # arXiv category
      max_results: 10
      days_back: 7
    active: true
    priority: "high"
```

**Configuration Options**:
- `category`: arXiv subject category (cs.AI, stat.ML, etc.)
- `keywords`: Search term list
- `max_results`: Papers per check
- `days_back`: Search time window

### URL Sources

Direct web content fetching:

```yaml
sources:
  - type: "url"
    name: "Important Research Paper"
    config:
      url: "https://example.com/paper.pdf"
      metadata:
        category: "neuroscience"
        priority: "high"
    active: true
    priority: "high"
```

### File Sources

Local file system monitoring:

```yaml
sources:
  - type: "file"
    name: "Research Papers Directory"
    config:
      path: "/data/research/papers"
      pattern: "*.pdf"  # File glob pattern
      recursive: true
    active: true
    priority: "medium"
```

### RSS Feed Sources

News and blog content:

```yaml
sources:
  - type: "rss"
    name: "AI News"
    config:
      url: "https://example.com/rss.xml"
      fetch_interval_hours: 6
      max_articles: 10
    active: true
    priority: "medium"
```

---

## Adding Custom Sources

### Method 1: Configuration-Based Extension

Add new sources through `research_sources.yaml`:

```yaml
sources:
  # Custom API source
  - type: "api"
    name: "Company Research Database"
    config:
      base_url: "https://api.company.com/research"
      api_key: "${COMPANY_API_KEY}"
      endpoint: "/papers"
      params:
        status: "published"
        limit: 50
    active: true
    priority: "high"

  # Custom database source
  - type: "database"
    name: "Internal Research DB"
    config:
      connection_string: "${INTERNAL_DB_URL}"
      query: "SELECT * FROM research_papers WHERE published_date > ?"
      params: ["2024-01-01"]
    active: false
    priority: "medium"
```

### Method 2: Code-Based Extension

Implement custom source classes:

```python
from research_agent.research import SourceDiscovery
from typing import List, Dict, Any
import aiohttp

class CustomAPISource:
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config['base_url']
        self.api_key = config['api_key']
        
    async def discover_content(self) -> List[Dict]:
        """Discover new content from custom API."""
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            async with session.get(f"{self.base_url}/papers", headers=headers) as response:
                data = await response.json()
                
                content_items = []
                for item in data['papers']:
                    content_items.append({
                        'url': item['pdf_url'],
                        'title': item['title'],
                        'metadata': {
                            'source': 'custom_api',
                            'api_id': item['id'],
                            'published_date': item['published_date']
                        }
                    })
                return content_items

# Register custom source
source_discovery = SourceDiscovery()
source_discovery.register_source_type('custom_api', CustomAPISource)
```

---

## Implementing Source Connectors

### Source Connector Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SourceConnector(ABC):
    """Abstract base class for source connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    async def discover_content(self) -> List[Dict]:
        """Discover new content items."""
        pass
        
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate source connectivity."""
        pass
        
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get source metadata."""
        pass
```

### Example: GitHub Repository Source

```python
import aiohttp
from datetime import datetime, timedelta

class GitHubSource(SourceConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config.get('token')
        self.repos = config.get('repos', [])
        self.days_back = config.get('days_back', 7)
        self.labels = config.get('labels', ['research', 'paper'])
        
    async def discover_content(self) -> List[Dict]:
        """Discover research content from GitHub."""
        content_items = []

        for repo in self.repos:
            repo_items = await self._search_repo(repo)
            content_items.extend(repo_items)

        return content_items

    async def _search_repo(self, repo: str) -> List[Dict]:
        """Search a specific GitHub repository."""
        since = (datetime.now() - timedelta(days=self.days_back)).isoformat()
        
        # Search for issues with research labels
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            'state': 'all',
            'since': since,
            'labels': ','.join(self.labels)
        }
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f"GitHub API error for {repo}: {response.status}")
                    return []

                issues = await response.json()

                content_items = []
                for issue in issues:
                    # Extract paper URLs from issue body
                    body = issue.get('body', '')
                    paper_urls = self._extract_paper_urls(body)

                    for paper_url in paper_urls:
                        content_items.append({
                            'url': paper_url,
                            'title': f"{issue['title']} - {repo}",
                            'content': body,
                            'metadata': {
                                'source': 'github',
                                'repo': repo,
                                'issue_number': issue['number'],
                                'issue_url': issue['html_url'],
                                'labels': [label['name'] for label in issue.get('labels', [])],
                                'author': issue['user']['login'] if issue.get('user') else None
                            }
                        })

                return content_items
                
    def _extract_paper_urls(self, text: str) -> List[str]:
        """Extract arXiv and PDF URLs from text."""
        import re

        urls = []

        # arXiv URLs
        arxiv_patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
            r'arxiv\.org/(?:abs|pdf)/(\w+/\d+)'
        ]

        for pattern in arxiv_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                urls.append(f"https://arxiv.org/pdf/{match}.pdf")

        # Direct PDF URLs
        pdf_matches = re.findall(r'https?://[^\s]+\.pdf', text, re.IGNORECASE)
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
                    headers['Authorization'] = f'token {self.token}'

                # Test with first repo
                test_repo = self.repos[0]
                async with session.get(f"https://api.github.com/repos/{test_repo}", headers=headers) as response:
                    return response.status == 200
        except:
            return False
            
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'github',
            'repos': self.repos,
            'description': f'GitHub repositories: {", ".join(self.repos)}'
        }

# Usage in configuration
sources:
  - type: "github"
    name: "ML Research Repository"
    config:
      repos: ["papers-we-love/papers-we-love"]
      token: "${GITHUB_TOKEN}"  # Optional
      days_back: 14
    active: true
    priority: "medium"
```

### Example: Semantic Scholar Source

```python
class SemanticScholarSource(SourceConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.queries = config.get('queries', [])
        self.limit = config.get('limit', 10)
        self.min_citation_count = config.get('min_citation_count', 0)
        
    async def discover_content(self) -> List[Dict]:
        """Discover papers from Semantic Scholar."""
        content_items = []

        for query in self.queries:
            query_items = await self._search_query(query)
            content_items.extend(query_items)

        return content_items

    async def _search_query(self, query: str) -> List[Dict]:
        """Search Semantic Scholar for a specific query."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': self.limit,
            'fields': 'title,abstract,authors,year,citationCount,externalIds,venue'
        }
        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    print(f"Semantic Scholar API error: {response.status}")
                    return []

                data = await response.json()

                content_items = []
                for paper in data.get('data', []):
                    # Skip papers with too few citations
                    if paper.get('citationCount', 0) < self.min_citation_count:
                        continue

                    # Get PDF URL from external IDs
                    pdf_url = self._get_pdf_url(paper.get('externalIds', {}))

                    if pdf_url:
                        content_items.append({
                            'url': pdf_url,
                            'title': paper.get('title', ''),
                            'content': paper.get('abstract', ''),
                            'metadata': {
                                'source': 'semantic_scholar',
                                'paper_id': paper.get('paperId'),
                                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                                'year': paper.get('year'),
                                'citation_count': paper.get('citationCount', 0),
                                'venue': paper.get('venue'),
                                'query': query
                            }
                        })

                return content_items

    def _get_pdf_url(self, external_ids: Dict[str, Any]) -> str:
        """Extract PDF URL from external IDs."""
        # Try arXiv first
        if 'ArXiv' in external_ids:
            return f"https://arxiv.org/pdf/{external_ids['ArXiv']}.pdf"

        # Try DOI
        if 'DOI' in external_ids:
            doi = external_ids['DOI']
            return f"https://doi.org/{doi}"

        return None

    async def validate_connection(self) -> bool:
        """Validate Semantic Scholar API access."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers['x-api-key'] = self.api_key

                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={'query': 'test', 'limit': 1},
                    headers=headers
                ) as response:
                    return response.status == 200
        except:
            return False
            
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'semantic_scholar',
            'queries': self.queries,
            'description': f'Semantic Scholar queries: {", ".join(self.queries)}'
        }
```

---

## Content Format Support

### Extending Content Extractors

Add support for new content formats:

```python
from research_agent.research import ContentExtractor
import docx

class ExtendedContentExtractor(ContentExtractor):
    def extract_text(self, content: str, content_type: str = None) -> str:
        """Extract text from various formats including DOCX."""
        if content_type == 'docx' or content.endswith('.docx'):
            return self._extract_from_docx(content)
        else:
            return super().extract_text(content, content_type)
            
    def _extract_from_docx(self, content_bytes) -> str:
        """Extract text from DOCX files."""
        doc = docx.Document(io.BytesIO(content_bytes))
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
```

### Custom Content Types

```yaml
ingestion:
  content_types:
    - type: "docx"
      max_size_mb: 10
      enabled: true
    - type: "latex"
      max_size_mb: 5
      enabled: true
    - type: "jupyter"
      max_size_mb: 20
      enabled: true
```

---

## Rate Limiting and Throttling

### Source-Level Rate Limiting

```yaml
research:
  rate_limit:
    requests_per_minute: 30
    concurrent_requests: 5
    backoff_seconds: 2
    
  sources:
    - type: "api"
      name: "Rate Limited API"
      config:
        rate_limit:
          requests_per_hour: 100  # Source-specific limits
          burst_limit: 10
```

### Implementing Custom Rate Limiting

```python
import asyncio
from collections import deque
import time

class TokenBucketRateLimiter:
    def __init__(self, rate_per_second: float, burst_size: int):
        self.rate = rate_per_second
        self.burst = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                await asyncio.sleep(0.1)  # Wait before retrying

# Usage in source connector
class RateLimitedSource(SourceConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        rate_config = config.get('rate_limit', {})
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_second=rate_config.get('rate_per_second', 1.0),
            burst_size=rate_config.get('burst_size', 5)
        )
        
    async def discover_content(self) -> List[Dict]:
        await self.rate_limiter.acquire()
        # ... rest of discovery logic
```

---

## Testing Custom Sources

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_github_source():
    config = {
        'repos': ['test/repo'],
        'token': 'fake_token',
        'days_back': 1
    }
    
    source = GitHubSource(config)
    
    # Mock HTTP response
    mock_response = {
        'data': [
            {
                'html_url': 'https://github.com/test/repo/issues/1',
                'title': 'Research Paper Discussion',
                'body': 'Check out this paper about transformers...',
                'number': 1,
                'labels': [{'name': 'research'}, {'name': 'paper'}]
            }
        ]
    }
    
    # Test discovery
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        content = await source.discover_content()
        
        assert len(content) == 1
        assert content[0]['title'] == 'Research Paper Discussion'
        assert content[0]['metadata']['source'] == 'github'
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_source_integration():
    """Test full source integration with task queue."""
    from research_agent.orchestrator import TaskQueue
    from research_agent.research import ManualSourceManager
    
    # Setup
    queue = TaskQueue(config)
    await queue.initialize()
    
    manager = ManualSourceManager(queue)
    
    # Add custom source
    task_id = await manager.add_url_source(
        url="https://example.com/custom-paper.pdf",
        metadata={"custom_field": "test_value"}
    )
    
    # Verify task created
    task = await queue.get_next_task()
    assert task is not None
    assert task.payload['url'] == "https://example.com/custom-paper.pdf"
    assert task.payload['metadata']['custom_field'] == "test_value"
```

### Load Testing

```python
import asyncio
import time

async def load_test_source(source, num_requests=100):
    """Load test a source connector."""
    start_time = time.time()
    
    tasks = []
    for i in range(num_requests):
        tasks.append(source.discover_content())
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    print(f"Load test results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Requests/sec: {num_requests/duration:.2f}")
    
    return results
```

---

## Best Practices

### Source Design

1. **Idempotent Operations**: Multiple calls should not create duplicates
2. **Error Resilience**: Handle network failures gracefully
3. **Rate Limiting**: Respect source API limits
4. **Incremental Updates**: Only fetch new content since last check

### Content Quality

1. **Metadata Enrichment**: Add rich metadata for better processing
2. **Content Validation**: Verify content before adding to queue
3. **Duplicate Detection**: Avoid processing duplicate content
4. **Format Consistency**: Normalize content formats

### Performance Optimization

1. **Batch Operations**: Process multiple items together
2. **Caching**: Cache expensive operations
3. **Async Processing**: Use async/await throughout
4. **Resource Limits**: Set reasonable timeouts and limits

### Monitoring and Maintenance

1. **Health Checks**: Implement connection validation
2. **Metrics Collection**: Track success rates and performance
3. **Error Logging**: Comprehensive error reporting
4. **Configuration Updates**: Support runtime reconfiguration

### Security Considerations

1. **API Key Management**: Secure storage of credentials
2. **Input Validation**: Validate all external inputs
3. **Rate Limiting**: Prevent abuse of external APIs
4. **Audit Logging**: Track all source interactions

### Example: Complete Custom Source Implementation

```python
# custom_sources.py
from research_agent.research import SourceConnector
import aiohttp
from typing import List, Dict, Any
from datetime import datetime, timedelta

class IEEE_XploreSource(SourceConnector):
    """IEEE Xplore research paper source."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.query = config['query']
        self.max_results = config.get('max_results', 10)
        
    async def discover_content(self) -> List[Dict]:
        """Discover papers from IEEE Xplore."""
        base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            'apikey': self.api_key,
            'querytext': self.query,
            'max_records': self.max_results,
            'start_year': start_date.year,
            'sort_order': 'desc',
            'sort_field': 'publication_date'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"IEEE API error: {response.status}")
                    
                data = await response.json()
                
                content_items = []
                for article in data.get('articles', []):
                    # Get PDF URL if available
                    pdf_url = article.get('pdf_url')
                    if not pdf_url:
                        continue
                        
                    content_items.append({
                        'url': pdf_url,
                        'title': article.get('title', ''),
                        'metadata': {
                            'source': 'ieee_xplore',
                            'article_number': article.get('article_number'),
                            'publication_date': article.get('publication_date'),
                            'authors': [author.get('full_name', '') for author in article.get('authors', [])],
                            'abstract': article.get('abstract', ''),
                            'doi': article.get('doi', '')
                        }
                    })
                    
                return content_items
                
    async def validate_connection(self) -> bool:
        """Validate IEEE Xplore API access."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://ieeexploreapi.ieee.org/api/v1/search/articles",
                    params={'apikey': self.api_key, 'querytext': 'test', 'max_records': 1}
                ) as response:
                    return response.status == 200
        except:
            return False
            
    def get_source_info(self) -> Dict[str, Any]:
        return {
            'type': 'ieee_xplore',
            'query': self.query,
            'description': f'IEEE Xplore search for: {self.query}'
        }

# Registration
def register_custom_sources(source_discovery):
    """Register all custom source types."""
    source_discovery.register_source_type('ieee_xplore', IEEE_XploreSource)
    source_discovery.register_source_type('github', GitHubSource)
    source_discovery.register_source_type('semantic_scholar', SemanticScholarSource)
```

This comprehensive guide shows how to extend the Autonomous Research Agent with custom research sources, from simple configuration additions to full custom connector implementations.

---

**Last Updated**: January 14, 2026