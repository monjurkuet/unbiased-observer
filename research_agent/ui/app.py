import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime, timedelta
import random
import time

# Page configuration
st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    .status-online {
        color: #10b981;
        font-weight: 500;
    }
    .status-offline {
        color: #ef4444;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Mock data functions
def get_mock_stats():
    return {
        'entities': 15420,
        'relationships': 45280,
        'communities': 234,
        'papers_processed': 1250,
        'active_tasks': 8,
        'uptime': '7d 14h 32m'
    }

def get_mock_graph_data():
    # Create a simple knowledge graph
    nodes = []
    edges = []
    
    # Add different types of nodes
    node_types = ['concept', 'field', 'paper', 'researcher']
    colors = {'concept': '#3b82f6', 'field': '#10b981', 'paper': '#f59e0b', 'researcher': '#8b5cf6'}
    
    for i in range(50):
        node_type = random.choice(node_types)
        nodes.append({
            'id': f'node_{i}',
            'label': f'{node_type.capitalize()} {i}',
            'type': node_type,
            'color': colors[node_type],
            'size': random.randint(10, 30)
        })
    
    # Create random connections
    for i in range(80):
        source = random.randint(0, 49)
        target = random.randint(0, 49)
        if source != target:
            edges.append({
                'source': f'node_{source}',
                'target': f'node_{target}',
                'weight': random.randint(1, 5)
            })
    
    return nodes, edges

def create_network_graph(nodes, edges):
    # Create networkx graph
    G = nx.Graph()
    
    for node in nodes:
        G.add_node(node['id'], 
                  label=node['label'], 
                  type=node['type'], 
                  color=node['color'],
                  size=node['size'])
    
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Calculate positions
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{G.nodes[node]['label']}<br>Type: {G.nodes[node]['type']}")
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            line_color='white'
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))
    
    return fig

# Sidebar
def create_sidebar():
    st.sidebar.title("🧠 Research Agent")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Knowledge Graph", "Query Interface", "Analytics"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Status
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        st.markdown("**Status:**")
    with col2:
        st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown(f"**Uptime:** {get_mock_stats()['uptime']}")
    st.sidebar.markdown(f"**Active Tasks:** {get_mock_stats()['active_tasks']}")
    
    return page

# Main content
def main():
    page = create_sidebar()
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Knowledge Graph":
        show_graph_view()
    elif page == "Query Interface":
        show_query_interface()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    st.markdown('<h1 class="main-header">Research Agent Dashboard</h1>', unsafe_allow_html=True)
    
    # Stats cards
    stats = get_mock_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Knowledge Entities", f"{stats['entities']:,}", "+12%")
    
    with col2:
        st.metric("Relationships", f"{stats['relationships']:,}", "+8%")
    
    with col3:
        st.metric("Research Communities", stats['communities'], "+15%")
    
    with col4:
        st.metric("Papers Processed", f"{stats['papers_processed']:,}", "+25%")
    
    st.markdown("---")
    
    # Recent activity and active tasks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Activity")
        activities = [
            ("Processed arXiv paper", "Deep Learning for Scientific Discovery", "2 min ago"),
            ("Added research source", "Nature Neuroscience article", "15 min ago"),
            ("Generated community", "Neural Network architectures", "1 hour ago"),
            ("Completed analysis", "Citation network analysis", "2 hours ago")
        ]
        
        for action, detail, time_ago in activities:
            st.markdown(f"**{action}**")
            st.caption(f"{detail} • {time_ago}")
            st.markdown("")
    
    with col2:
        st.subheader("⚡ Active Tasks")
        tasks = [
            ("Processing arXiv papers", 75, "running"),
            ("Community detection", 45, "running"),
            ("Citation analysis", 90, "running"),
            ("Knowledge extraction", 30, "queued")
        ]
        
        for task_name, progress, status in tasks:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"**{task_name}**")
                st.progress(progress / 100)
            with col_b:
                if status == "running":
                    st.markdown('<span style="color: #10b981;">● Running</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: #6b7280;">○ Queued</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Add Research Source", use_container_width=True):
            st.info("Upload papers or URLs functionality would go here")
    
    with col2:
        if st.button("🔍 Start Query", use_container_width=True):
            st.info("Navigate to Query Interface")
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("Navigate to Analytics page")

def show_graph_view():
    st.markdown('<h1 class="main-header">Knowledge Graph Explorer</h1>', unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search nodes", placeholder="Enter node name...")
    
    with col2:
        node_types = st.multiselect(
            "Filter by type",
            ["concept", "field", "paper", "researcher"],
            default=["concept", "field", "paper", "researcher"],
            label_visibility="collapsed"
        )
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Generate mock data
    nodes, edges = get_mock_graph_data()
    
    # Apply filters
    filtered_nodes = [n for n in nodes if n['type'] in node_types]
    if search_term:
        filtered_nodes = [n for n in filtered_nodes 
                         if search_term.lower() in n['label'].lower()]
    
    # Filter edges to only include filtered nodes
    node_ids = set(n['id'] for n in filtered_nodes)
    filtered_edges = [e for e in edges 
                     if e['source'] in node_ids and e['target'] in node_ids]
    
    # Create and display graph
    if filtered_nodes:
        fig = create_network_graph(filtered_nodes, filtered_edges)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Showing {len(filtered_nodes)} nodes and {len(filtered_edges)} connections**")
    else:
        st.info("No nodes match the current filters. Try adjusting your search or filters.")

def show_query_interface():
    st.markdown('<h1 class="main-header">Research Query Interface</h1>', unsafe_allow_html=True)
    
    # Query input
    st.markdown("### Ask questions about research, papers, concepts, or connections")
    
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the main research areas in machine learning?",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "What are the main research areas in machine learning?",
            "Find papers about neural networks from 2024",
            "Show connections between deep learning and computer vision",
            "Who are the top researchers in natural language processing?"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.query = example
                st.rerun()
    
    # Execute query
    if search_button and query.strip():
        with st.spinner("Searching knowledge graph..."):
            time.sleep(2)  # Simulate processing
            
            # Mock results
            results = [
                {
                    "type": "concept",
                    "name": "Machine Learning",
                    "description": "A subset of AI that enables computers to learn from data",
                    "connections": 45,
                    "papers": 1250
                },
                {
                    "type": "field", 
                    "name": "Supervised Learning",
                    "description": "ML with labeled training data",
                    "connections": 23,
                    "papers": 890
                },
                {
                    "type": "field",
                    "name": "Unsupervised Learning", 
                    "description": "ML without labeled training data",
                    "connections": 18,
                    "papers": 567
                }
            ]
            
            st.success(f"Found {len(results)} results in 1.2 seconds")
            
            # Display results
            for result in results:
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Type badge
                        type_colors = {
                            "concept": "#3b82f6",
                            "field": "#10b981", 
                            "paper": "#f59e0b",
                            "researcher": "#8b5cf6"
                        }
                        st.markdown(
                            f'<span style="background-color: {type_colors.get(result["type"], "#6b7280")}; '
                            'color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">'
                            f'{result["type"].title()}</span>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(f"**{result['name']}**")
                        st.write(result['description'])
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"🔗 {result['connections']} connections")
                        with col_b:
                            st.caption(f"📄 {result['papers']} papers")
                    
                    st.divider()

def show_analytics():
    st.markdown('<h1 class="main-header">Research Analytics</h1>', unsafe_allow_html=True)
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 days", "Last 30 days", "Last 3 months", "Last year"],
        index=0
    )
    
    # Mock analytics data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    papers_processed = [random.randint(5, 25) for _ in range(30)]
    entities_added = [random.randint(20, 100) for _ in range(30)]
    
    df = pd.DataFrame({
        'date': dates,
        'papers': papers_processed,
        'entities': entities_added
    })
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Papers Processed Over Time")
        fig1 = px.line(df, x='date', y='papers', 
                      title="Daily Paper Processing",
                      labels={'papers': 'Papers', 'date': 'Date'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📊 Entity Growth")
        fig2 = px.area(df, x='date', y='entities',
                      title="Daily Entity Additions", 
                      labels={'entities': 'Entities', 'date': 'Date'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Research areas breakdown
    st.subheader("🎯 Research Areas Distribution")
    
    research_areas = pd.DataFrame({
        'area': ['AI/ML', 'Computer Vision', 'NLP', 'Robotics', 'Other'],
        'papers': [450, 280, 320, 150, 50],
        'percentage': [45, 28, 32, 15, 5]
    })
    
    fig3 = px.pie(research_areas, values='papers', names='area', 
                 title="Papers by Research Area")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Top researchers
    st.subheader("👥 Top Researchers by Publications")
    
    researchers = pd.DataFrame({
        'researcher': ['Dr. Smith', 'Dr. Johnson', 'Dr. Chen', 'Dr. Garcia', 'Dr. Kim'],
        'publications': [45, 38, 32, 28, 25],
        'citations': [1250, 980, 750, 620, 580]
    })
    
    st.dataframe(researchers, use_container_width=True)

if __name__ == "__main__":
    main()
