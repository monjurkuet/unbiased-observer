# Research Agent Web Interface

A lightweight Streamlit-based web interface for the Autonomous Research Agent.

## Features

- **📊 Dashboard**: Real-time statistics and monitoring
- **🕸️ Knowledge Graph**: Interactive network visualization
- **🔍 Query Interface**: Natural language research queries
- **📈 Analytics**: Research trends and insights

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interface
python run.py
# or directly:
streamlit run app.py
```

Access at: http://localhost:8501

## Architecture

- **Streamlit**: Frontend framework for data applications
- **Plotly**: Interactive charts and network graphs
- **NetworkX**: Graph algorithms and layout
- **Mock Data**: Simulated research agent data (ready for API integration)

## Development

The interface is designed to be:
- **Lightweight**: Minimal dependencies, fast startup
- **Extensible**: Easy to add new pages and features
- **Responsive**: Works on desktop and mobile
- **Real-time**: Ready for live data updates

## Integration

To connect to the actual research agent:

1. Replace mock data functions with API calls
2. Add authentication and user management
3. Implement real-time WebSocket connections
4. Add data export and sharing features

## File Structure

```
streamlit-ui/
├── app.py              # Main Streamlit application
├── run.py              # Launch script
├── requirements.txt    # Dependencies
└── README.md          # This file
```
