# Research API

Content discovery, fetching, and extraction components.

---

## Table of Contents

- [Overview](#overview)
- [ArxivIntegrator](#arxivintegrator)
- [ContentFetcher](#contentfetcher)
- [ContentExtractor](#contentextractor)
- [ManualSourceManager](#manualsourcemanager)
- [SourceDiscovery](#sourcediscovery)
- [Usage Examples](#usage-examples)

---

## Overview

The research module handles content discovery from various sources, asynchronous content fetching with rate limiting, and text extraction from multiple formats. It supports both automated discovery (arXiv monitoring) and manual source addition.

### Key Components

- **ArxivIntegrator**: arXiv API integration for automated paper discovery
- **ContentFetcher**: Asynchronous content retrieval with rate limiting
- **ContentExtractor**: Text extraction from HTML, PDF, Markdown, plain text
- **ManualSourceManager**: Manual source addition interface
- **SourceDiscovery**: Automated source discovery coordination

---

## ArxivIntegrator

arXiv API wrapper for automated research paper discovery.

### Class Signature

```python
class ArxivIntegrator:
    def __init__(self, config: Config = None, logger: logging.Logger = None)
    async def search_by_keywords(self, keywords: List[str], max_results: int = 10, days_back: int = 7) -> List[Dict]
    async def search_by_category(self, category: str, max_results: int = 10, days_back: int = 7) -> List[Dict]
    async def get_paper_details(self, paper_id: str) -> Dict
    async def discover_new_papers(self, search_configs: List[Dict]) -> List[Dict]
    async def add_paper_to_queue(self, paper: Dict, task_queue: TaskQueue) -> int
    async def run_monitoring_cycle(self, task_queue: TaskQueue) -> Dict
```

### Methods

#### search_by_keywords()

Search arXiv papers by keyword list.

```python
async def search_by_keywords(
    self,
    keywords: List[str],
    max_results: int = 10,
    days_back: int = 7
) -> List[Dict]:
    """Search papers by keywords within time window."""
```

**Parameters**:
- `keywords`: List of search terms (e.g., ["machine learning", "deep learning"])
- `max_results`: Maximum papers to return (default: 10)
- `days_back`: Number of days to search back from today (default: 7)

**Returns**: List of paper dictionaries with metadata

**Example Paper Dict**:
```python
{
    "id": "2301.07041",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "abstract": "The dominant sequence transduction models...",
    "categories": ["cs.CL", "cs.AI"],
    "published": "2023-01-17",
    "updated": "2023-01-17",
    "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
    "primary_category": "cs.CL"
}
```

#### search_by_category()

Search papers by arXiv category.

```python
async def search_by_category(
    self,
    category: str,
    max_results: int = 10,
    days_back: int = 7
) -> List[Dict]:
    """Search papers by arXiv category."""
```

**Parameters**:
- `category`: arXiv category (e.g., "cs.AI", "cs.LG", "stat.ML")
- `max_results`: Maximum papers to return
- `days_back`: Days to search back

**Common Categories**:
- `cs.AI`: Artificial Intelligence
- `cs.LG`: Machine Learning
- `cs.CL`: Computation and Language
- `cs.CV`: Computer Vision
- `stat.ML`: Statistics/Machine Learning

#### get_paper_details()

Get detailed information for a specific paper.

```python
async def get_paper_details(self, paper_id: str) -> Dict:
    """Get full paper metadata and abstract."""
```

**Parameters**:
- `paper_id`: arXiv paper ID (e.g., "2301.07041")

**Returns**: Complete paper metadata including full abstract

#### discover_new_papers()

Discover new papers based on search configurations.

```python
async def discover_new_papers(self, search_configs: List[Dict]) -> List[Dict]:
    """Discover new papers from search configurations."""
```

**Parameters**:
- `search_configs`: List of search configuration dictionaries

**Config Format**:
```python
{
    "name": "AI Research",
    "keywords": ["artificial intelligence", "machine learning"],
    "max_results": 5,
    "days_back": 7,
    "active": true
}
```

#### add_paper_to_queue()

Add discovered paper to task queue for processing.

```python
async def add_paper_to_queue(self, paper: Dict, task_queue: TaskQueue) -> int:
    """Add paper to fetch queue. Returns task ID."""
```

#### run_monitoring_cycle()

Run complete monitoring cycle for all configured searches.

```python
async def run_monitoring_cycle(self, task_queue: TaskQueue) -> Dict:
    """Run monitoring cycle and queue new papers."""
```

**Returns**: Monitoring statistics

```python
{
    "papers_discovered": 15,
    "tasks_created": 12,
    "errors": 0,
    "duration_seconds": 45.2
}
```

---

## ArxivSourceManager

Manages automated arXiv monitoring and task creation.

### Class Signature

```python
class ArxivSourceManager:
    def __init__(self, config: Config, logger: logging.Logger)
    async def initialize(self) -> None
    async def run_monitoring_cycle(self) -> Dict
```

### Methods

#### initialize()

Initialize with configuration and task queue.

```python
async def initialize(self) -> None:
    """Initialize with config and task queue."""
```

#### run_monitoring_cycle()

Execute monitoring cycle for configured arXiv searches.

```python
async def run_monitoring_cycle(self) -> Dict:
    """Monitor arXiv and create fetch tasks."""
```

---

## ContentFetcher

Asynchronous content fetching with rate limiting and error handling.

### Class Signature

```python
class ContentFetcher:
    def __init__(self, config: Config = None, logger: logging.Logger = None)
    async def fetch_url(self, url: str, timeout: int = 30) -> str
    async def fetch_file(self, file_path: str) -> str
    async def fetch_batch(self, urls: List[str]) -> Dict[str, str]
    async def _rate_limit_wait(self) -> None
```

### Methods

#### fetch_url()

Fetch content from URL with rate limiting.

```python
async def fetch_url(self, url: str, timeout: int = 30) -> str:
    """Fetch content from URL asynchronously."""
```

**Parameters**:
- `url`: HTTP/HTTPS URL to fetch
- `timeout`: Request timeout in seconds (default: 30)

**Returns**: Raw content as string

**Features**:
- Automatic HTTPS upgrading
- Rate limiting with configurable delays
- Retry logic with exponential backoff
- User-agent headers for API compatibility

#### fetch_file()

Read content from local file.

```python
async def fetch_file(self, file_path: str) -> str:
    """Read content from local file."""
```

**Parameters**:
- `file_path`: Absolute path to local file

**Supported Formats**:
- Plain text files
- PDF documents (text extraction)
- Markdown files
- HTML files

#### fetch_batch()

Fetch multiple URLs concurrently with rate limiting.

```python
async def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
    """Fetch multiple URLs concurrently."""
```

**Parameters**:
- `urls`: List of URLs to fetch

**Returns**: Dictionary mapping URLs to content strings

**Concurrency Control**:
- Respects rate limits across batch
- Semaphore-based concurrency control
- Error isolation (one failure doesn't stop batch)

#### _rate_limit_wait()

Internal rate limiting mechanism.

```python
async def _rate_limit_wait(self) -> None:
    """Wait to respect rate limits."""
```

**Configuration**:
```yaml
research:
  rate_limit:
    requests_per_minute: 30
    concurrent_requests: 5
    backoff_seconds: 2
```

### Exceptions

#### FetchError

Raised when content fetching fails.

```python
class FetchError(Exception):
    """Raised when content fetching fails."""
    def __init__(self, url: str, status_code: int = None, message: str = None):
        self.url = url
        self.status_code = status_code
        self.message = message
```

---

## ContentExtractor

Text extraction and normalization from various content formats.

### Class Signature

```python
class ContentExtractor:
    def __init__(self, config: Config = None, logger: logging.Logger = None)
    def extract_text(self, content: str, content_type: str = None) -> str
    def _detect_content_type(self, content: str) -> str
    def _extract_from_html(self, content: str) -> str
    def _extract_from_markdown(self, content: str) -> str
    def _extract_from_plain_text(self, content: str) -> str
    def truncate_to_max_length(self, content: str, max_length: int = None) -> str
```

### Methods

#### extract_text()

Extract clean text from various formats.

```python
def extract_text(self, content: str, content_type: str = None) -> str:
    """Extract clean text from content."""
```

**Parameters**:
- `content`: Raw content string
- `content_type`: Content type hint ("html", "markdown", "pdf", "text")

**Returns**: Clean, normalized text

**Auto-Detection**:
- HTML: `<html>`, `<!DOCTYPE html>` detection
- Markdown: `#`, `##`, `**bold**` pattern detection
- PDF: Binary PDF header detection
- Plain text: Default fallback

#### _extract_from_html()

Extract text from HTML content.

```python
def _extract_from_html(self, content: str) -> str:
    """Extract text from HTML, removing scripts and styles."""
```

**Processing**:
- Remove `<script>` and `<style>` tags
- Extract text from remaining HTML
- Clean up whitespace and formatting
- Preserve paragraph structure

#### _extract_from_markdown()

Process Markdown content.

```python
def _extract_from_markdown(self, content: str) -> str:
    """Process Markdown, preserving structure."""
```

**Processing**:
- Remove Markdown formatting syntax
- Preserve headers and lists as text
- Convert links to descriptive text
- Maintain paragraph breaks

#### _extract_from_plain_text()

Process plain text content.

```python
def _extract_from_plain_text(self, content: str) -> str:
    """Clean and normalize plain text."""
```

**Processing**:
- Normalize whitespace
- Remove excessive line breaks
- Clean up encoding issues
- Preserve paragraph structure

#### truncate_to_max_length()

Limit content length for processing.

```python
def truncate_to_max_length(self, content: str, max_length: int = None) -> str:
    """Truncate content to maximum length."""
```

**Configuration**:
```yaml
ingestion:
  max_content_length: 100000  # characters
```

---

## ManualSourceManager

Interface for manually adding research sources.

### Class Signature

```python
class ManualSourceManager:
    def __init__(self, task_queue: TaskQueue, source_discovery: SourceDiscovery, logger: logging.Logger = None)
    async def add_url_source(self, url: str, metadata: Dict = None, priority: str = "medium") -> int
    async def add_file_source(self, file_path: str, metadata: Dict = None, priority: str = "medium") -> int
    async def add_text_source(self, text: str, metadata: Dict = None, priority: str = "medium") -> int
```

### Methods

#### add_url_source()

Add URL source for fetching.

```python
async def add_url_source(
    self,
    url: str,
    metadata: Dict = None,
    priority: str = "medium"
) -> int:
    """Add URL source. Returns task ID."""
```

**Parameters**:
- `url`: HTTP/HTTPS URL to research content
- `metadata`: Additional metadata for the source
- `priority`: Task priority ("high", "medium", "low")

**Returns**: Task ID for tracking

#### add_file_source()

Add local file source.

```python
async def add_file_source(
    self,
    file_path: str,
    metadata: Dict = None,
    priority: str = "medium"
) -> int:
    """Add file source. Returns task ID."""
```

**Parameters**:
- `file_path`: Absolute path to local file
- `metadata`: Source metadata
- `priority`: Task priority

#### add_text_source()

Add direct text content.

```python
async def add_text_source(
    self,
    text: str,
    metadata: Dict = None,
    priority: str = "medium"
) -> int:
    """Add text content directly. Returns task ID."""
```

**Parameters**:
- `text`: Raw text content
- `metadata`: Content metadata
- `priority`: Task priority

**Use Cases**:
- Pasted research notes
- API responses
- Manual content entry

---

## SourceDiscovery

Coordinates automated source discovery from configured feeds.

### Class Signature

```python
class SourceDiscovery:
    def __init__(self, config: Config, logger: logging.Logger = None)
    async def discover_new_content(self) -> List[Dict]
    async def add_manual_source(self, source_data: Dict) -> int
    def _load_sources(self) -> List[Dict]
    def get_sources(self) -> List[Dict]
    def get_active_sources(self) -> List[Dict]
```

### Methods

#### discover_new_content()

Discover new content from all configured sources.

```python
async def discover_new_content(self) -> List[Dict]:
    """Discover new content from configured sources."""
```

**Returns**: List of new content items

#### add_manual_source()

Add manually specified source.

```python
async def add_manual_source(self, source_data: Dict) -> int:
    """Add manual source. Returns source ID."""
```

#### _load_sources()

Load sources from configuration.

```python
def _load_sources(self) -> List[Dict]:
    """Load sources from research_sources.yaml."""
```

#### get_sources()

Get all configured sources.

```python
def get_sources(self) -> List[Dict]:
    """Get all sources configuration."""
```

#### get_active_sources()

Get only active sources.

```python
def get_active_sources(self) -> List[Dict]:
    """Get sources marked as active."""
```

---

## Usage Examples

### arXiv Paper Discovery

```python
from research_agent.research import ArxivIntegrator

# Initialize integrator
integrator = ArxivIntegrator()

# Search by keywords
papers = await integrator.search_by_keywords(
    keywords=["machine learning", "neural networks"],
    max_results=5,
    days_back=3
)

for paper in papers:
    print(f"{paper['title']} by {', '.join(paper['authors'])}")

# Search by category
ai_papers = await integrator.search_by_category(
    category="cs.AI",
    max_results=10,
    days_back=7
)
```

### Content Fetching

```python
from research_agent.research import ContentFetcher

fetcher = ContentFetcher()

# Fetch single URL
content = await fetcher.fetch_url("https://arxiv.org/pdf/2301.07041.pdf")

# Fetch from local file
content = await fetcher.fetch_file("/path/to/research_paper.pdf")

# Batch fetch multiple URLs
urls = [
    "https://example.com/paper1.pdf",
    "https://example.com/paper2.pdf"
]
results = await fetcher.fetch_batch(urls)

for url, content in results.items():
    print(f"Fetched {len(content)} characters from {url}")
```

### Content Extraction

```python
from research_agent.research import ContentExtractor

extractor = ContentExtractor()

# Extract from HTML
html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"
text = extractor.extract_text(html_content, "html")
print(text)  # "Title Content"

# Extract from Markdown
md_content = "# Title\n\nThis is **bold** text."
text = extractor.extract_text(md_content, "markdown")
print(text)  # "Title This is bold text."

# Auto-detection
mixed_content = "<html><body>HTML content</body></html>"
text = extractor.extract_text(mixed_content)  # Auto-detects HTML
```

### Manual Source Addition

```python
from research_agent.research import ManualSourceManager

manager = ManualSourceManager(task_queue, source_discovery)

# Add URL source
task_id = await manager.add_url_source(
    url="https://www.nature.com/articles/s41586-023-12345-6",
    metadata={"category": "neuroscience", "priority": "high"},
    priority="high"
)

# Add local file
task_id = await manager.add_file_source(
    file_path="/data/research/my_paper.pdf",
    metadata={"author": "Dr. Smith", "conference": "ICML 2023"}
)

# Add direct text
task_id = await manager.add_text_source(
    text="Recent research shows that...",
    metadata={"source": "notes", "topic": "AI ethics"}
)
```

### Automated Monitoring Setup

```python
from research_agent.research import ArxivSourceManager

# Initialize with config
source_manager = ArxivSourceManager(config, logger)
await source_manager.initialize()

# Run monitoring cycle (called by scheduler every 2 hours)
stats = await source_manager.run_monitoring_cycle()
print(f"Discovered {stats['papers_discovered']} new papers")
```

---

## Configuration

### arXiv Integration

```yaml
research:
  arxiv:
    monitoring_enabled: true
    monitoring_interval_hours: 2
    search_configs:
      - name: "AI Research"
        keywords: ["artificial intelligence", "machine learning"]
        max_results: 5
        days_back: 7
        active: true
      - name: "Computer Vision"
        category: "cs.CV"
        max_results: 3
        days_back: 3
        active: true
```

### Content Fetching

```yaml
research:
  rate_limit:
    requests_per_minute: 30
    concurrent_requests: 5
    backoff_seconds: 2
  timeout: 30  # seconds
```

### Content Processing

```yaml
ingestion:
  max_content_length: 100000
  content_types:
    - type: "pdf"
      enabled: true
    - type: "html"
      enabled: true
    - type: "markdown"
      enabled: true
    - type: "text"
      enabled: true
```

---

## Monitoring

### Discovery Metrics

- **Papers Discovered**: New papers found per monitoring cycle
- **Sources Processed**: URLs/files processed successfully
- **Fetch Success Rate**: Percentage of successful content fetches
- **Content Size Distribution**: Average content length by type

### Performance Metrics

- **Fetch Latency**: Average time to retrieve content
- **Extraction Time**: Time to process different content types
- **Error Rates**: Failures by content type and source
- **Rate Limit Hits**: Number of rate limit delays

---

## Troubleshooting

### arXiv API Issues

**Problem**: No papers returned from search

**Solutions**:
```python
# Check API connectivity
papers = await integrator.search_by_keywords(["test"], max_results=1)
print(f"API working: {len(papers)} results")

# Verify search terms
papers = await integrator.search_by_keywords(["machine learning"], max_results=5)
print(f"Keyword search: {len(papers)} results")

# Try category search
papers = await integrator.search_by_category("cs.AI", max_results=5)
print(f"Category search: {len(papers)} results")
```

### Content Fetching Issues

**Problem**: HTTP errors or timeouts

**Solutions**:
```bash
# Test URL accessibility
curl -I "https://example.com/paper.pdf"

# Check SSL certificates
curl -v "https://example.com/paper.pdf" 2>&1 | grep -A 5 "SSL certificate"

# Adjust timeout
content = await fetcher.fetch_url(url, timeout=60)

# Check rate limiting
# Reduce requests_per_minute in config
```

### Content Extraction Issues

**Problem**: Poor text quality from PDFs/HTML

**Solutions**:
```python
# Check content type detection
content_type = extractor._detect_content_type(raw_content)
print(f"Detected type: {content_type}")

# Manual type specification
text = extractor.extract_text(raw_content, content_type="html")

# Verify content length
print(f"Content length: {len(raw_content)}")
if len(raw_content) > 100000:
    print("Content too long, consider chunking")
```

### Source Addition Issues

**Problem**: Tasks not created

**Solutions**:
```python
# Verify task queue
pending = await task_queue.get_pending_count()
print(f"Pending tasks: {pending}")

# Check task creation
task_id = await manager.add_url_source("https://example.com")
print(f"Created task: {task_id}")

# Verify task in database
tasks = await task_queue.get_last_failed_tasks(5)
for task in tasks:
    print(f"Task {task.id}: {task.status} - {task.error_message}")
```

---

## Performance Tuning

### arXiv Optimization

- **Batch Searches**: Combine related keywords
- **Category Filtering**: Use specific categories over broad keywords
- **Time Windows**: Balance recency vs. volume

### Fetching Optimization

- **Concurrent Limits**: Adjust based on network capacity
- **Rate Limiting**: Respect source API limits
- **Caching**: Cache frequently accessed content

### Extraction Optimization

- **Content Type Hints**: Provide content_type when known
- **Length Limits**: Truncate very long content
- **Batch Processing**: Process multiple items together

---

**Last Updated**: January 14, 2026