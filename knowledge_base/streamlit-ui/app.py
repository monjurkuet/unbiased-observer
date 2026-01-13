"""
Streamlit UI for the Knowledge Base GraphRAG system
Connects to the FastAPI backend for real-time knowledge graph exploration
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")
WS_URL = os.getenv("STREAMLIT_WS_URL", "ws://localhost:8000/ws")

st.set_page_config(
    page_title="Knowledge Base Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .node-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .edge-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def create_network_graph(nodes: List[Dict], edges: List[Dict]) -> go.Figure:
    """Create an interactive network graph visualization"""
    G = nx.Graph()

    for node in nodes:
        G.add_node(node["id"], **node)

    for edge in edges:
        G.add_edge(edge["source"], edge["target"], **edge)

    pos = nx.spring_layout(G, k=1, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = G.nodes[node]
        node_text.append(
            f"{node_data.get('name', node)}<br>Type: {node_data.get('type', 'Unknown')}"
        )
        node_type = node_data.get("type", "Unknown")
        color_map = {
            "Person": "#FF6B6B",
            "Organization": "#4ECDC4",
            "Event": "#45B7D1",
            "Concept": "#96CEB4",
            "Location": "#FFEAA7",
        }
        node_color.append(color_map.get(node_type, "#BDC3C7"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[G.nodes[node].get("name", str(node)) for node in G.nodes()],
        textposition="top center",
        marker=dict(showscale=False, color=node_color, size=20, line_width=2),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Knowledge Graph Visualization", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


class APIClient:
    """Client for interacting with the Knowledge Base API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            response = requests.get(f"{self.base_url}/api/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get stats: {e}")
            return {}

    def get_nodes(
        self, limit: int = 100, node_type: Optional[str] = None
    ) -> List[Dict]:
        """Get nodes from the knowledge graph"""
        try:
            params = {"limit": limit}
            if node_type:
                params["node_type"] = node_type
            response = requests.get(f"{self.base_url}/api/nodes", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get nodes: {e}")
            return []

    def get_edges(self, limit: int = 500) -> List[Dict]:
        """Get edges from the knowledge graph"""
        try:
            response = requests.get(
                f"{self.base_url}/api/edges", params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get edges: {e}")
            return []

    def get_communities(self) -> List[Dict]:
        """Get community information"""
        try:
            response = requests.get(f"{self.base_url}/api/communities")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get communities: {e}")
            return []

    def search_nodes(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search nodes using text query"""
        try:
            response = requests.post(
                f"{self.base_url}/api/search", json={"query": query, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            return {"results": [], "count": 0}

    def get_graph_data(self, limit: int = 500) -> Dict[str, Any]:
        """Get graph data for visualization"""
        try:
            response = requests.get(
                f"{self.base_url}/api/graph", params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get graph data: {e}")
            return {"nodes": [], "edges": []}

    def ingest_text(
        self, text: str, filename: str = "streamlit_upload.txt"
    ) -> Dict[str, Any]:
        """Ingest text content"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ingest/text",
                json={"text": text, "filename": filename},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            return {"status": "error", "message": str(e)}

    def detect_communities(self) -> Dict[str, Any]:
        """Run community detection"""
        try:
            response = requests.post(f"{self.base_url}/api/community/detect")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Community detection failed: {e}")
            return {"status": "error", "message": str(e)}

    def run_summarization(self) -> Dict[str, Any]:
        """Run recursive summarization"""
        try:
            response = requests.post(f"{self.base_url}/api/summarize")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            return {"status": "error", "message": str(e)}


def main():
    """Main Streamlit application"""

    api_client = APIClient()

    st.markdown(
        '<h1 class="main-header">🧠 Knowledge Base Explorer</h1>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("📊 Dashboard")

        if st.button("🔄 Refresh Stats"):
            with st.spinner("Loading statistics..."):
                stats = api_client.get_stats()

            if stats:
                st.markdown("### Database Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nodes", stats.get("nodes_count", 0))
                    st.metric("Edges", stats.get("edges_count", 0))
                with col2:
                    st.metric("Communities", stats.get("communities_count", 0))
                    st.metric("Events", stats.get("events_count", 0))

        st.header("⚙️ Operations")
        if st.button("🔍 Detect Communities"):
            with st.spinner("Running community detection..."):
                result = api_client.detect_communities()
                if result.get("status") == "success":
                    st.success(
                        result.get("message", "Communities detected successfully")
                    )
                else:
                    st.error("Community detection failed")

        if st.button("📝 Run Summarization"):
            with st.spinner("Running summarization..."):
                result = api_client.run_summarization()
                if result.get("status") == "success":
                    st.success("Summarization completed successfully")
                else:
                    st.error("Summarization failed")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Overview", "🔍 Search", "📄 Ingest", "🕸️ Graph", "🏘️ Communities"]
    )

    with tab1:
        st.header("Knowledge Base Overview")

        col1, col2, col3, col4 = st.columns(4)
        stats = api_client.get_stats()

        with col1:
            st.metric("Total Nodes", stats.get("nodes_count", 0))
        with col2:
            st.metric("Total Edges", stats.get("edges_count", 0))
        with col3:
            st.metric("Communities", stats.get("communities_count", 0))
        with col4:
            st.metric("Events", stats.get("events_count", 0))

        st.subheader("Recent Nodes")
        nodes = api_client.get_nodes(limit=10)
        for node in nodes[:5]:
            st.markdown(
                f"""
            <div class="node-card">
                <strong>{node["name"]}</strong> ({node["type"]})<br>
                <small>{node["description"][:100]}...</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab2:
        st.header("Search Knowledge Base")

        query = st.text_input("Search query", placeholder="Enter search terms...")
        limit = st.slider("Results limit", 5, 50, 10)

        if st.button("🔍 Search") and query:
            with st.spinner("Searching..."):
                results = api_client.search_nodes(query, limit)

            st.subheader(f"Search Results ({results.get('count', 0)} found)")

            for result in results.get("results", []):
                with st.expander(f"{result['name']} ({result['type']})"):
                    st.write(f"**Description:** {result['description']}")
                    st.write(f"**ID:** {result['id']}")

    with tab3:
        st.header("Ingest New Content")

        text_input = st.text_area(
            "Enter text to ingest",
            height=200,
            placeholder="Paste your text content here...",
        )

        uploaded_file = st.file_uploader("Or upload a text file", type=["txt", "md"])

        if st.button("🚀 Ingest Content"):
            content = None
            filename = "streamlit_upload.txt"

            if text_input.strip():
                content = text_input
            elif uploaded_file:
                content = uploaded_file.read().decode("utf-8")
                filename = uploaded_file.name

            if content:
                with st.spinner("Ingesting content..."):
                    result = api_client.ingest_text(content, filename)

                if result.get("status") == "success":
                    st.success(result.get("message", "Content ingested successfully"))
                    st.rerun()
                else:
                    st.error("Ingestion failed")
            else:
                st.warning("Please enter text or upload a file")

    with tab4:
        st.header("Knowledge Graph Visualization")

        if st.button("🔄 Load Graph"):
            with st.spinner("Loading graph data..."):
                graph_data = api_client.get_graph_data(limit=200)

            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            if nodes:
                fig = create_network_graph(nodes, edges)
                st.plotly_chart(fig, use_container_width=True)

                st.info(f"Displaying {len(nodes)} nodes and {len(edges)} edges")

                node_types = defaultdict(int)
                for node in nodes:
                    node_types[node.get("type", "Unknown")] += 1

                st.subheader("Node Types")
                for node_type, count in node_types.items():
                    st.write(f"**{node_type}:** {count}")
            else:
                st.info("No graph data available. Try ingesting some content first.")

    with tab5:
        st.header("Community Analysis")

        communities = api_client.get_communities()

        if communities:
            st.subheader(f"Found {len(communities)} Communities")

            for community in communities:
                with st.expander(f"Community {community['id']}: {community['title']}"):
                    st.write(f"**Nodes:** {community['node_count']}")
                    if community["summary"]:
                        st.write(f"**Summary:** {community['summary']}")
        else:
            st.info(
                "No communities detected yet. Run community detection from the sidebar."
            )


if __name__ == "__main__":
    main()
