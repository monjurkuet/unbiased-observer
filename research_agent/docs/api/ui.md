# UI API

Streamlit-based web interface for knowledge graph visualization and querying.

---

## Table of Contents

- [Overview](#overview)
- [Main Application](#main-application)
- [Dashboard Functions](#dashboard-functions)
- [Knowledge Graph View](#knowledge-graph-view)
- [Query Interface](#query-interface)
- [Analytics View](#analytics-view)
- [Usage Examples](#usage-examples)

---

## Overview

The UI module provides a Streamlit-based web interface for visualizing and interacting with the Autonomous Research Agent's knowledge graph. Currently implemented with mock data, it's designed for easy integration with the actual research agent backend.

### Key Components

- **Main Application**: Streamlit app with navigation
- **Dashboard**: Real-time statistics and system status
- **Knowledge Graph**: Interactive network visualization
- **Query Interface**: Natural language research queries
- **Analytics**: Charts and trends analysis

---

## Main Application

Core Streamlit application with page navigation.

### app.py Structure

```python
# Main application functions
def get_mock_stats() -> Dict[str, Any]
def get_mock_graph_data() -> Dict[str, Any]
def create_network_graph(data: Dict) -> plotly.graph_objects.Figure
def create_sidebar() -> None
def main() -> None

# Page functions
def show_dashboard() -> None
def show_graph_view() -> None
def show_query_interface() -> None
def show_analytics() -> None
```

### main()

Main application entry point with navigation.

```python
def main() -> None:
    """Main Streamlit application."""
```

**Features**:
- Page navigation sidebar
- Responsive layout
- Real-time data updates (mock)
- Error handling and user feedback

### create_sidebar()

Create navigation sidebar.

```python
def create_sidebar() -> None:
    """Create application sidebar with navigation."""
```

**Navigation Pages**:
- 📊 Dashboard
- 🕸️ Knowledge Graph
- 🔍 Query Interface
- 📈 Analytics

---

## Dashboard Functions

Real-time statistics and system monitoring display.

### show_dashboard()

Display main dashboard with key metrics.

```python
def show_dashboard() -> None:
    """Display main dashboard page."""
```

**Dashboard Sections**:
- **Key Metrics**: Entities, relationships, communities, papers processed
- **Recent Activity**: Latest tasks and operations
- **Active Tasks**: Currently running tasks
- **Quick Actions**: Manual triggers and controls

### get_mock_stats()

Generate mock statistics for dashboard.

```python
def get_mock_stats() -> Dict[str, Any]:
    """Get mock statistics for dashboard display."""
```

**Returns**: Statistics dictionary

```python
{
    "total_entities": 15420,
    "total_relationships": 28950,
    "total_communities": 45,
    "papers_processed": 890,
    "active_tasks": 3,
    "system_health": "healthy",
    "recent_activity": [
        {"timestamp": "2024-01-14T10:30:00Z", "action": "Processed community analysis", "status": "completed"},
        {"timestamp": "2024-01-14T10:25:00Z", "action": "Ingested research paper", "status": "completed"}
    ]
}
```

---

## Knowledge Graph View

Interactive network visualization of the knowledge graph.

### show_graph_view()

Display interactive knowledge graph visualization.

```python
def show_graph_view() -> None:
    """Display knowledge graph visualization page."""
```

**Features**:
- **Interactive Network**: Zoom, pan, node selection
- **Node Types**: Different colors for entities, concepts, people
- **Search & Filter**: Find specific nodes and relationships
- **Node Details**: Click nodes for detailed information
- **Layout Options**: Force-directed, hierarchical, circular layouts

### get_mock_graph_data()

Generate mock knowledge graph data.

```python
def get_mock_graph_data() -> Dict[str, Any]:
    """Get mock knowledge graph data for visualization."""
```

**Returns**: Graph data structure

```python
{
    "nodes": [
        {"id": 1, "label": "Transformer Architecture", "type": "concept", "size": 20},
        {"id": 2, "label": "Ashish Vaswani", "type": "person", "size": 15},
        {"id": 3, "label": "Attention Mechanism", "type": "concept", "size": 18}
    ],
    "edges": [
        {"source": 1, "target": 2, "type": "authored_by", "weight": 1.0},
        {"source": 1, "target": 3, "type": "uses", "weight": 0.8}
    ]
}
```

### create_network_graph()

Create Plotly network visualization.

```python
def create_network_graph(data: Dict) -> plotly.graph_objects.Figure:
    """Create interactive network graph visualization."""
```

**Visualization Features**:
- **Node Coloring**: By entity type (person, concept, organization)
- **Edge Styling**: Different line styles for relationship types
- **Interactive Tooltips**: Node and edge information on hover
- **Responsive Design**: Adapts to screen size

---

## Query Interface

Natural language querying against the knowledge graph.

### show_query_interface()

Display query interface for natural language questions.

```python
def show_query_interface() -> None:
    """Display query interface page."""
```

**Features**:
- **Natural Language Input**: Ask questions in plain English
- **Example Queries**: Pre-built query suggestions
- **Query History**: Previous queries and results
- **Result Display**: Formatted answers with sources
- **Follow-up Questions**: Related query suggestions

### Query Examples

**Supported Query Types**:
- "What papers has author X written?"
- "What are the main concepts in field Y?"
- "Who collaborates with researcher Z?"
- "What are the latest developments in topic A?"
- "Show me relationships between concepts B and C"

### Mock Query Responses

```python
# Example query response structure
{
    "query": "What are the main concepts in natural language processing?",
    "results": [
        {
            "answer": "The main concepts include transformer architectures, attention mechanisms, and self-supervised learning.",
            "confidence": 0.92,
            "sources": ["Paper A", "Paper B", "Survey C"],
            "related_queries": [
                "What is a transformer architecture?",
                "How does attention work in NLP?",
                "What is self-supervised learning?"
            ]
        }
    ],
    "processing_time": 1.2
}
```

---

## Analytics View

Charts and trends analysis for research data.

### show_analytics()

Display analytics dashboard with charts and trends.

```python
def show_analytics() -> None:
    """Display analytics page with charts and trends."""
```

**Analytics Sections**:
- **Paper Processing Over Time**: Daily/weekly ingestion rates
- **Entity Growth**: Accumulation of entities and relationships
- **Research Area Distribution**: Pie charts of topic coverage
- **Top Researchers**: Most prolific authors in the graph
- **Community Evolution**: How communities change over time
- **Citation Networks**: Most cited papers and authors

### Chart Types

- **Line Charts**: Time series for growth metrics
- **Bar Charts**: Top researchers, popular concepts
- **Pie Charts**: Research area distribution
- **Network Charts**: Citation and collaboration networks
- **Heatmaps**: Research activity by time and topic

---

## Usage Examples

### Running the UI

```python
# From the ui directory
cd ui
python run.py

# Or directly with Streamlit
streamlit run app.py --server.port 8501
```

### Accessing the Interface

```bash
# Open in browser
open http://localhost:8501

# Or use curl for API testing
curl http://localhost:8501/health
```

### Integration with Research Agent

```python
# Future integration pattern
from research_agent.monitoring import MetricsCollector
from research_agent.ingestion import IngestionPipeline

# Replace mock functions with real data
def get_real_stats():
    metrics = MetricsCollector(config)
    return await metrics.get_summary_metrics()

def get_real_graph_data():
    # Query knowledge graph database
    # Return nodes and edges for visualization
    pass

# In app.py
stats = get_real_stats()  # Instead of get_mock_stats()
graph_data = get_real_graph_data()  # Instead of get_mock_graph_data()
```

### Customizing the UI

```python
# Add new page
def show_new_page():
    st.title("Custom Research Analytics")
    # Add custom visualizations
    pass

# Add to navigation
pages = {
    "Dashboard": show_dashboard,
    "Knowledge Graph": show_graph_view,
    "Query Interface": show_query_interface,
    "Analytics": show_analytics,
    "Custom Page": show_new_page  # New page
}
```

### Adding Real-time Updates

```python
# Add auto-refresh
import time

def auto_refresh():
    while True:
        # Update data
        stats = get_real_stats()
        # Update UI components
        time.sleep(30)  # Refresh every 30 seconds
```

---

## Configuration

### UI Configuration

```yaml
ui:
  port: 8501
  host: "0.0.0.0"
  debug: false
  theme: "light"  # light, dark, auto
  max_upload_size_mb: 50
```

### Streamlit Configuration

```toml
# .streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#FF4B4B"
```

### Integration Configuration

```yaml
ui:
  integration:
    enabled: true
    api_base_url: "http://localhost:8000"  # Future API endpoint
    refresh_interval_seconds: 30
    cache_timeout_seconds: 300
```

---

## UI Architecture

### Component Structure

```
ui/
├── app.py              # Main Streamlit application
├── run.py              # Launch script with environment setup
├── requirements.txt    # UI dependencies
└── static/             # Static assets (future)
    ├── css/
    ├── js/
    └── images/
```

### Data Flow

```
User Request → Streamlit App → Data Functions → Database/API → Response → UI Update
```

### State Management

- **Session State**: User preferences and session data
- **Cache**: Expensive computations and API responses
- **Real-time Updates**: Periodic data refresh
- **Error Handling**: Graceful failure with user feedback

---

## Future Integration Points

### API Integration

```python
# Planned API endpoints
class ResearchAgentAPI:
    @staticmethod
    async def get_stats() -> Dict:
        """Get real-time system statistics."""
        pass

    @staticmethod
    async def query_graph(query: str) -> Dict:
        """Execute natural language query."""
        pass

    @staticmethod
    async def get_graph_data(filters: Dict = None) -> Dict:
        """Get knowledge graph data for visualization."""
        pass

    @staticmethod
    async def get_analytics(time_range: str = "7d") -> Dict:
        """Get analytics data."""
        pass
```

### Real-time Updates

```python
# WebSocket integration for live updates
import asyncio
import websockets

async def websocket_handler(websocket, path):
    while True:
        # Send real-time updates
        stats = await get_real_stats()
        await websocket.send(json.dumps(stats))
        await asyncio.sleep(30)
```

### Authentication

```python
# Future authentication
def check_authentication():
    if not st.session_state.authenticated:
        st.error("Please log in to access the research agent.")
        return False
    return True

# Add to each page
if not check_authentication():
    return
```

---

## Performance Considerations

### UI Optimization

- **Lazy Loading**: Load data only when needed
- **Caching**: Cache expensive computations
- **Pagination**: Handle large datasets
- **Compression**: Minimize data transfer

### Visualization Performance

- **Node Limiting**: Limit visible nodes for performance
- **Level of Detail**: Simplify complex graphs
- **Progressive Loading**: Load graph in chunks
- **WebGL Rendering**: Hardware-accelerated graphics

### Memory Management

- **Data Cleanup**: Clear unused data from memory
- **Session Limits**: Limit concurrent sessions
- **Resource Monitoring**: Track memory usage
- **Auto-scaling**: Scale UI instances as needed

---

## Troubleshooting

### Streamlit Not Starting

**Problem**: UI fails to start

**Symptoms**:
- Port 8501 not accessible
- Streamlit errors

**Solutions**:

```bash
# Check Streamlit installation
streamlit --version

# Test basic Streamlit app
cd ui
python3 -c "import streamlit as st; print('Streamlit OK')"

# Start with debug
streamlit run app.py --logger.level debug

# Check port availability
netstat -tlnp | grep 8501
```

### Data Loading Issues

**Problem**: UI shows no data or mock data

**Symptoms**:
- Empty graphs
- Mock statistics displayed

**Solutions**:

```python
# Check database connection in UI
cd ui
python3 -c "
from research_agent.config import load_config
config = load_config()
print('Config loaded')
# Test database connection
"

# Verify data exists
psql -U research_agent -d knowledge_graph -c "
SELECT COUNT(*) as nodes FROM nodes;
SELECT COUNT(*) as edges FROM edges;
SELECT COUNT(*) as communities FROM communities;
"
```

### Visualization Issues

**Problem**: Network graph not rendering

**Symptoms**:
- No graph displayed
- Plotly errors

**Solutions**:

```python
# Check Plotly installation
import plotly.graph_objects as go
print("Plotly available")

# Test graph creation
fig = create_network_graph(get_mock_graph_data())
print(f"Graph created with {len(fig.data)} traces")
```

### Performance Issues

**Problem**: UI slow or unresponsive

**Symptoms**:
- Long loading times
- Browser timeouts

**Solutions**:

```python
# Enable caching
@st.cache_data(ttl=300)
def load_cached_stats():
    return get_mock_stats()

# Reduce data size
graph_data = get_mock_graph_data()
if len(graph_data['nodes']) > 500:
    graph_data['nodes'] = graph_data['nodes'][:500]  # Limit nodes
```

---

## Development

### Adding New Pages

```python
def show_new_page():
    st.title("New Research Page")
    st.write("Custom research functionality")
    
    # Add page logic here
    pass

# Update navigation in main()
pages["New Page"] = show_new_page
```

### Customizing Styling

```python
# Custom CSS
st.markdown("""
<style>
.custom-class {
    color: #FF4B4B;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)
```

### Testing the UI

```python
# Run tests
pytest ui/tests/ -v

# Manual testing
streamlit run app.py --server.headless true --server.port 8502
```

---

## Deployment

### Local Development

```bash
# Run locally
cd ui
python run.py
```

### Production Deployment

```bash
# Use systemd service
sudo cp deployment/research-agent-ui.service /etc/systemd/system/
sudo systemctl enable research-agent-ui
sudo systemctl start research-agent-ui
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

**Last Updated**: January 14, 2026