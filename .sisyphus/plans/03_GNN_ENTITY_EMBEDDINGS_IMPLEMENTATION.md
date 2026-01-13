# GNN ENTITY EMBEDDINGS - IMPLEMENTATION PLAN

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Implementation-Ready
**Priority**: P0
**Estimated Effort**: 8 weeks

---

## 1. EXECUTIVE SUMMARY

**Objective**: Enhance entity embeddings by incorporating graph structural information using Graph Neural Networks (GNNs), improving entity resolution, link prediction, and overall knowledge base quality.

**Key Deliverables**:
- Heterogeneous Graph Attention Network (GAT) model trained on knowledge graph
- Enhanced entity embeddings combining text + structural features
- Integrated entity resolution using GNN embeddings
- 12%+ improvement in entity resolution F1 score

**Impact**: High - Core improvement affecting all downstream tasks

---

## 2. TECHNICAL APPROACH

### 2.1 Architecture Selection: Heterogeneous GAT

**Rationale**:
- **Attention Mechanism**: Learns importance of different neighbors (vs. uniform aggregation)
- **Heterogeneous Support**: Handles multiple edge types (AUTHORED, LEADS, PART_OF, etc.)
- **Interpretability**: Attention weights provide explainability
- **Proven Performance**: State-of-the-art for knowledge graphs

**Model Diagram**:
```
Input: Entity (text embedding: 768D)
    │
    ▼
┌─────────────────────────────────┐
│  Relation-Type Attention Layer   │
│  ── AUTHORIZED neighbors ──►│
│  ── LEADS neighbors ───────►│
│  ── PART_OF neighbors ─────►│
│  ── COLLABORATES neighbors ──►│
└─────────────────────────────────┘
    │
    ▼
  Multi-head aggregation (4 heads)
    │
    ▼
  GNN Output (128D)
    │
    ▼
┌─────────────────────────────────┐
│   Projection Layer           │
│   (GNN embedding → Final)   │
└─────────────────────────────────┘
    │
    ▼
Final Embedding (768D)
```

### 2.2 Loss Functions

**Multi-Task Learning Objective**:

```python
L_total = α * L_link + β * L_recon + γ * L_community

Where:
- α = 1.0 (Link prediction - primary task)
- β = 0.5 (Feature reconstruction - regularization)
- γ = 0.3 (Community alignment - structural preservation)
```

---

## 3. PHASE 1: SETUP & BASELINE (WEEKS 1-2)

### 3.1 Week 1: Environment Setup

**Tasks**:
1. Install dependencies
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add torch-geometric
uv add sentence-transformers
uv add scikit-learn
```

2. Create project structure
```
knowledge_base/
├── gnn/
│   ├── __init__.py
│   ├── models.py           # GNN architectures
│   ├── data.py            # Data loading/prep
│   ├── training.py        # Training loops
│   ├── evaluation.py      # Metrics
│   └── export.py         # Export embeddings to DB
```

3. Set up GPU environment (if available)
```python
# gnn/__init__.py
import torch

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    return device

DEVICE = setup_device()
```

**Deliverable**: Working development environment with GPU support

### 3.2 Week 2: Data Export & Baseline

**Tasks**:

1. Export knowledge graph from PostgreSQL
```python
# gnn/data.py
import psycopg
from torch_geometric.data import HeteroData
import numpy as np

def export_knowledge_graph(db_conn_str: str) -> HeteroData:
    """Export complete knowledge graph for GNN training"""

    data = HeteroData()

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # Load nodes
            cur.execute("SELECT id, embedding FROM nodes")
            node_rows = cur.fetchall()

            node_ids = [str(row[0]) for row in node_rows]
            node_features = np.array([row[1] for row in node_rows])

            # Create mapping
            node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

            # Load edges by type
            edge_types = {}
            cur.execute("SELECT DISTINCT type FROM edges")
            types = [row[0] for row in cur.fetchall()]

            for edge_type in types:
                cur.execute(
                    """
                    SELECT source_id, target_id, weight
                    FROM edges
                    WHERE type = %s
                    """,
                    (edge_type,)
                )
                edge_rows = cur.fetchall()

                if edge_rows:
                    source_indices = [node_to_idx[str(r[0])] for r in edge_rows]
                    target_indices = [node_to_idx[str(r[1])] for r in edge_rows]
                    weights = np.array([r[2] for r in edge_rows])

                    edge_types[edge_type] = {
                        'edge_index': np.array([source_indices, target_indices]),
                        'edge_attr': weights
                    }

    data['entity'].x = torch.tensor(node_features, dtype=torch.float)
    for edge_type, edges in edge_types.items():
        data[edge_type].edge_index = torch.tensor(edges['edge_index'], dtype=torch.long)
        data[edge_type].edge_attr = torch.tensor(edges['edge_attr'], dtype=torch.float).unsqueeze(1)

    return data, node_to_idx
```

2. Load communities for evaluation
```python
def load_community_labels(db_conn_str: str, node_to_idx: dict) -> torch.Tensor:
    """Load community memberships for evaluation"""

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cm.node_id, c.id
                FROM community_membership cm
                JOIN communities c ON cm.community_id = c.id
                WHERE c.level = 0
                """
            )

            labels = {}
            for node_id, comm_id in cur.fetchall():
                labels[str(node_id)] = str(comm_id)

            # Create label tensor
            label_tensor = torch.zeros(len(node_to_idx), dtype=torch.long)
            for node_id, idx in node_to_idx.items():
                if node_id in labels:
                    label_tensor[idx] = labels[node_id]

            return label_tensor
```

3. Train baseline GCN model
```python
# gnn/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineGCN(nn.Module):
    """Simple GCN baseline for comparison"""

    def __init__(self, num_features, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_features)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
```

4. Establish baseline metrics
```python
# gnn/evaluation.py
from sklearn.metrics import silhouette_score, adjusted_rand_score

def evaluate_embeddings(embeddings: np.ndarray, community_labels: np.ndarray) -> dict:
    """Evaluate embedding quality"""

    # Silhouette score (cluster quality)
    silhouette = silhouette_score(embeddings, community_labels)

    # Intra-cluster vs inter-cluster distance
    intra_dist = compute_intra_cluster_distance(embeddings, community_labels)
    inter_dist = compute_inter_cluster_distance(embeddings, community_labels)

    return {
        'silhouette': silhouette,
        'intra_distance': intra_dist,
        'inter_distance': inter_dist
    }
```

**Deliverable**: Baseline GCN model with performance metrics

---

## 4. PHASE 2: GNN DEVELOPMENT (WEEKS 3-5)

### 4.1 Week 3: Heterogeneous GAT Implementation

**Task**: Implement multi-relation GAT model

```python
# gnn/models.py
from torch_geometric.nn import GATConv, HeteroConv
from typing import Dict

class HeterogeneousGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network
    Handles multiple edge types with attention
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        edge_types: list[str] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.edge_types = edge_types or ['AUTHORED', 'LEADS', 'PART_OF']
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # GAT convolutions for each edge type
        self.convs = nn.ModuleDict()
        for edge_type in self.edge_types:
            self.convs[edge_type] = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=1,  # Edge weight
                concat=True
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Dict[str, torch.Tensor],
        edge_weights: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)

        # Apply GAT for each edge type
        edge_embeddings = []

        for edge_type in self.edge_types:
            edge_index = edge_indices[edge_type]
            edge_attr = edge_weights.get(edge_type) if edge_weights else None

            # Apply GAT convolution
            out = self.convs[edge_type](x, edge_index, edge_attr)

            # Mean across heads
            out = out.mean(dim=1)

            # Residual connection
            edge_embeddings.append(out)

        # Aggregate across edge types (mean)
        aggregated = torch.stack(edge_embeddings).mean(dim=0)

        aggregated = self.dropout(aggregated)
        x = x + aggregated  # Residual

        # Project back to original dimension
        output = self.output_proj(x)

        return output

    def get_attention_weights(self, x, edge_indices, edge_type):
        """Extract attention weights for interpretation"""

        edge_index = edge_indices[edge_type]

        # Get attention from specific convolution
        conv = self.convs[edge_type]
        x_proj = self.input_proj(x)

        # Get attention scores
        out = conv(x_proj, edge_index, return_attention_weights=True)

        # Attention weights: (num_heads, num_edges)
        attention = out[1][0]

        return attention
```

**Deliverable**: Working heterogeneous GAT model

### 4.2 Week 4: Training Pipeline

**Task**: Implement multi-task training loop

```python
# gnn/training.py
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def generate_negative_edges(
    num_nodes: int,
    num_pos_edges: int,
    num_neg_per_pos: int = 1
) -> torch.Tensor:
    """Generate negative edges for link prediction"""

    all_possible_edges = set()
    neg_edges = []

    while len(neg_edges) < num_pos_edges * num_neg_per_pos:
        # Sample random pair
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)

        if node1 == node2:
            continue

        # Sort for consistency
        edge = tuple(sorted([node1, node2]))

        if edge not in all_possible_edges:
            all_possible_edges.add(edge)
            neg_edges.append([node1, node2])

    return torch.tensor(neg_edges, dtype=torch.long).t()

def train_gnn(
    model: nn.Module,
    data: HeteroData,
    community_labels: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.001,
    device: torch.device = None
) -> dict:
    """Train GNN with multi-task learning"""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Prepare training data
    all_edge_indices = torch.cat([
        data[edge_type].edge_index
        for edge_type in ['AUTHORED', 'LEADS', 'PART_OF']
        if edge_type in data
    ], dim=1)

    pos_edges = all_edge_indices
    neg_edges = generate_negative_edges(
        data['entity'].num_nodes,
        pos_edges.shape[1],
        num_neg_per_pos=1
    ).to(device)

    # Move to device
    x = data['entity'].x.to(device)
    edge_dict = {
        edge_type: {
            'edge_index': data[edge_type].edge_index.to(device),
            'edge_attr': data[edge_type].edge_attr.to(device) if edge_type in data else None
        }
        for edge_type in ['AUTHORED', 'LEADS', 'PART_OF']
        if edge_type in data
    }
    community_labels = community_labels.to(device)

    training_history = {
        'total_loss': [],
        'link_loss': [],
        'recon_loss': [],
        'community_loss': []
    }

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(x, edge_dict)

        # Compute losses
        link_loss = link_prediction_loss(embeddings, pos_edges, neg_edges)
        recon_loss = reconstruction_loss(embeddings, x)
        comm_loss = community_alignment_loss(embeddings, community_labels)

        # Weighted combination
        total_loss = 1.0 * link_loss + 0.5 * recon_loss + 0.3 * comm_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track history
        training_history['total_loss'].append(total_loss.item())
        training_history['link_loss'].append(link_loss.item())
        training_history['recon_loss'].append(recon_loss.item())
        training_history['community_loss'].append(comm_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Total={total_loss.item():.4f}, "
                  f"Link={link_loss.item():.4f}, "
                  f"Recon={recon_loss.item():.4f}, "
                  f"Comm={comm_loss.item():.4f}")

    return {
        'model': model,
        'embeddings': embeddings.detach().cpu(),
        'history': training_history
    }

# Loss functions

def link_prediction_loss(
    embeddings: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy for link prediction"""

    # Positive edge scores
    pos_head_emb = embeddings[pos_edges[0]]
    pos_tail_emb = embeddings[pos_edges[1]]
    pos_scores = (pos_head_emb * pos_tail_emb).sum(dim=1)
    pos_labels = torch.ones(pos_scores.shape[0], device=embeddings.device)

    # Negative edge scores
    neg_head_emb = embeddings[neg_edges[0]]
    neg_tail_emb = embeddings[neg_edges[1]]
    neg_scores = (neg_head_emb * neg_tail_emb).sum(dim=1)
    neg_labels = torch.zeros(neg_scores.shape[0], device=embeddings.device)

    # Combine
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    return F.binary_cross_entropy_with_logits(scores, labels)

def reconstruction_loss(
    embeddings: torch.Tensor,
    original_features: torch.Tensor
) -> torch.Tensor:
    """MSE loss for feature reconstruction"""

    reconstructed = model.output_proj(embeddings) if hasattr(model, 'output_proj') else embeddings
    return F.mse_loss(reconstructed, original_features)

def community_alignment_loss(
    embeddings: torch.Tensor,
    community_labels: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """Maximize intra-community similarity, minimize inter-community"""

    # Pairwise similarities (batch this for efficiency)
    sim_matrix = torch.mm(embeddings, embeddings.t())

    # Create masks
    same_community = (community_labels.unsqueeze(0) == community_labels.unsqueeze(1))
    different_community = ~same_community

    # Within-community similarities
    intra_sim = sim_matrix[same_community].mean() if same_community.sum() > 0 else torch.tensor(0.0)

    # Between-community similarities
    inter_sim = sim_matrix[different_community].mean() if different_community.sum() > 0 else torch.tensor(0.0)

    # Hinge loss
    return F.relu(margin + inter_sim - intra_sim)
```

**Deliverable**: Complete training pipeline with multi-task loss

### 4.3 Week 5: Hyperparameter Optimization

**Task**: Grid search over key hyperparameters

```python
# gnn/training.py
from itertools import product

def hyperparameter_search(
    data: HeteroData,
    community_labels: torch.Tensor,
    device: torch.device
) -> dict:
    """Grid search over hyperparameters"""

    # Hyperparameter grid
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'num_heads': [2, 4, 8],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [0.001, 0.0005, 0.0001]
    }

    results = []

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params in combinations:
        print(f"\nTesting: {params}")

        # Create model
        model = HeterogeneousGAT(
            num_features=data['entity'].x.shape[1],
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            dropout=params['dropout']
        )

        # Train
        result = train_gnn(
            model=model,
            data=data,
            community_labels=community_labels,
            epochs=50,  # Short epochs for search
            lr=params['lr'],
            device=device
        )

        # Evaluate
        metrics = evaluate_model(result['embeddings'], community_labels, data)

        results.append({
            'params': params,
            'final_loss': result['history']['total_loss'][-1],
            'metrics': metrics
        })

    # Find best
    best = min(results, key=lambda x: x['final_loss'])

    return best, results

def evaluate_model(
    embeddings: torch.Tensor,
    community_labels: torch.Tensor,
    data: HeteroData
) -> dict:
    """Evaluate model performance"""

    embeddings_np = embeddings.numpy()
    labels_np = community_labels.numpy()

    # Silhouette score
    silhouette = silhouette_score(embeddings_np, labels_np)

    # Link prediction accuracy (hold-out set)
    link_auc = evaluate_link_prediction_auc(embeddings, data)

    return {
        'silhouette': silhouette,
        'link_auc': link_auc
    }
```

**Deliverable**: Optimized hyperparameters

---

## 5. PHASE 3: EVALUATION (WEEK 6)

### 5.1 Entity Resolution Evaluation

**Task**: Evaluate impact on entity resolution F1 score

```python
# gnn/evaluation.py
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_entity_resolution(
    gnn_embeddings: dict,  # entity_id -> embedding
    test_pairs: list  # [(entity1, entity2, should_match)]
) -> dict:
    """Evaluate GNN embeddings for entity resolution"""

    predictions = []
    actuals = []

    for entity1_id, entity2_id, should_match in test_pairs:
        # Get embeddings
        emb1 = torch.tensor(gnn_embeddings[entity1_id])
        emb2 = torch.tensor(gnn_embeddings[entity2_id])

        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=0)

        # Decision (threshold 0.8)
        predicted_match = similarity > 0.8

        predictions.append(predicted_match)
        actuals.append(should_match)

    # Compute metrics
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def create_entity_resolution_test_set(db_conn_str: str) -> list:
    """Create test set for entity resolution evaluation"""

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # Get resolved pairs (same entity, different names)
            cur.execute(
                """
                SELECT n1.name, n2.name, n1.type
                FROM nodes n1
                JOIN nodes n2 ON n1.id = n2.id AND n1.name != n2.name
                LIMIT 50
                """
            )
            same_entity_pairs = [(row[0], row[1], True, row[2]) for row in cur.fetchall()]

            # Get different entity pairs
            cur.execute(
                """
                SELECT n1.name, n2.name, n1.type
                FROM nodes n1, nodes n2
                WHERE n1.id < n2.id
                AND n1.type = n2.type
                ORDER BY RANDOM()
                LIMIT 50
                """
            )
            diff_entity_pairs = [(row[0], row[1], False, row[2]) for row in cur.fetchall()]

    return same_entity_pairs + diff_entity_pairs
```

**Deliverable**: Entity resolution F1 score comparison

### 5.2 Link Prediction Evaluation

**Task**: Evaluate link prediction performance

```python
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_link_prediction_auc(
    embeddings: torch.Tensor,
    data: HeteroData,
    test_ratio: float = 0.2
) -> float:
    """Evaluate link prediction AUC"""

    # Split edges into train/test
    all_edge_indices = torch.cat([
        data[edge_type].edge_index
        for edge_type in ['AUTHORED', 'LEADS', 'PART_OF']
        if edge_type in data
    ], dim=1)

    # Shuffle and split
    perm = torch.randperm(all_edge_indices.shape[1])
    split_idx = int(all_edge_indices.shape[1] * (1 - test_ratio))

    train_edges = all_edge_indices[:, perm[:split_idx]]
    test_edges = all_edge_indices[:, perm[split_idx:]]

    # Generate negative edges for test
    neg_test_edges = generate_negative_edges(
        data['entity'].num_nodes,
        test_edges.shape[1],
        num_neg_per_pos=1
    )

    # Predict scores
    def predict_score(edge):
        head_emb = embeddings[edge[0]]
        tail_emb = embeddings[edge[1]]
        return (head_emb * tail_emb).sum().item()

    # Positive scores
    pos_scores = [predict_score(test_edges[:, i]) for i in range(test_edges.shape[1])]

    # Negative scores
    neg_scores = [predict_score(neg_test_edges[:, i]) for i in range(neg_test_edges.shape[1])]

    # Compute AUC
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    auc = roc_auc_score(labels, scores)

    return auc
```

**Deliverable**: Link prediction AUC score

### 5.3 Qualitative Analysis

**Task**: Manual inspection of nearest neighbors

```python
# gnn/evaluation.py
from sklearn.metrics.pairwise import cosine_similarity

def analyze_nearest_neighbors(
    gnn_embeddings: np.ndarray,
    entity_names: list[str],
    num_neighbors: int = 5
) -> dict:
    """Find and display nearest neighbors for analysis"""

    similarities = cosine_similarity(gnn_embeddings)

    results = {}

    for i, entity_name in enumerate(entity_names[:20]):  # Sample 20 entities
        # Get similarity scores
        sim_scores = similarities[i]

        # Get top-k (excluding self)
        top_indices = np.argsort(sim_scores)[-num_neighbors-1:-1][::-1]

        results[entity_name] = [
            (entity_names[idx], sim_scores[idx])
            for idx in top_indices
        ]

    return results

def print_neighbor_analysis(results: dict):
    """Print analysis for review"""

    for entity, neighbors in results.items():
        print(f"\n{entity}:")
        for neighbor, sim in neighbors:
            print(f"  - {neighbor}: {sim:.3f}")
```

**Deliverable**: Qualitative analysis report

---

## 6. PHASE 4: INTEGRATION (WEEK 7-8)

### 6.1 Week 7: Export Embeddings

**Task**: Export GNN embeddings to PostgreSQL

```python
# gnn/export.py
import psycopg

def export_embeddings_to_db(
    gnn_embeddings: torch.Tensor,
    node_to_idx: dict,
    db_conn_str: str
):
    """Export GNN embeddings to database"""

    # Convert to numpy
    embeddings_np = gnn_embeddings.numpy()

    # Create reverse mapping
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # Add new column for GNN embeddings
            cur.execute(
                """
                ALTER TABLE nodes
                ADD COLUMN IF NOT EXISTS gnn_embedding vector(768)
                """
            )

            # Batch update
            for idx in range(len(embeddings_np)):
                node_id = idx_to_node[idx]
                embedding = embeddings_np[idx].tolist()

                cur.execute(
                    """
                    UPDATE nodes
                    SET gnn_embedding = %s
                    WHERE id = %s
                    """,
                    (embedding, node_id)
                )

            conn.commit()

            print(f"Exported {len(embeddings_np)} embeddings to database")
```

**Deliverable**: GNN embeddings in database

### 6.2 Week 8: Integration with Entity Resolution

**Task**: Update entity resolver to use GNN embeddings

```python
# knowledge_base/resolver.py (modified)
class EnhancedEntityResolver(EntityResolver):
    """Entity resolver using GNN embeddings"""

    def __init__(self, *args, use_gnn=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gnn = use_gnn

    async def find_candidates(
        self, entity_name: str, embedding: List[float], threshold: float = 0.70
    ) -> List[Dict]:
        """Find candidates using GNN embeddings if enabled"""

        async with await psycopg.AsyncConnection.connect(self.db_conn_str) as conn:
            async with conn.cursor() as cur:
                if self.use_gnn:
                    # Use GNN embeddings
                    await cur.execute(
                        """
                        SELECT id, name, type, description,
                               1 - (gnn_embedding <=> %s::vector) as similarity
                        FROM nodes
                        WHERE gnn_embedding IS NOT NULL
                          AND 1 - (gnn_embedding <=> %s::vector) > %s
                        ORDER BY similarity DESC
                        LIMIT 5
                        """,
                        (embedding, embedding, threshold)
                    )
                else:
                    # Use text embeddings (original)
                    await cur.execute(
                        """
                        SELECT id, name, type, description,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM nodes
                        WHERE 1 - (embedding <=> %s::vector) > %s
                        ORDER BY similarity DESC
                        LIMIT 5
                        """,
                        (embedding, embedding, threshold)
                    )

                async for row in cur:
                    candidates.append(
                        {
                            "id": str(row[0]),
                            "name": row[1],
                            "type": row[2],
                            "description": row[3],
                            "similarity": row[4],
                        }
                    )

        return candidates
```

**Deliverable**: Integrated entity resolution with GNN

---

## 7. TESTING & VALIDATION

### 7.1 Unit Tests

```python
# tests/test_gnn.py
import pytest
import torch
from gnn.models import HeterogeneousGAT

@pytest.fixture
def sample_graph():
    """Sample graph for testing"""

    data = {
        'entity': {
            'x': torch.randn(10, 768)
        },
        'AUTHORED': {
            'edge_index': torch.tensor([[0, 1, 2], [3, 4, 5]]),
            'edge_attr': torch.tensor([[1.0], [0.8], [0.9]])
        }
    }
    return data

def test_gat_forward_pass(sample_graph):
    """Test GAT forward pass"""

    model = HeterogeneousGAT(num_features=768, hidden_dim=128)

    x = sample_graph['entity']['x']
    edge_indices = {
        'AUTHORED': sample_graph['AUTHORED']['edge_index']
    }

    output = model(x, edge_indices)

    assert output.shape == (10, 768)  # Same as input

def test_attention_weights_shape(sample_graph):
    """Test attention weights extraction"""

    model = HeterogeneousGAT(num_features=768, hidden_dim=128, num_heads=4)

    x = sample_graph['entity']['x']
    edge_indices = {'AUTHORED': sample_graph['AUTHORED']['edge_index']}

    attention = model.get_attention_weights(x, edge_indices, 'AUTHORED')

    # Should be (num_heads, num_edges)
    assert attention.shape == (4, 3)
```

### 7.2 Integration Tests

```python
# tests/test_integration.py
import pytest
from knowledge_base.resolver import EnhancedEntityResolver

@pytest.mark.asyncio
async def test_gnn_entity_resolution(db_conn):
    """Test entity resolution with GNN embeddings"""

    resolver = EnhancedEntityResolver(
        db_conn_str=db_conn,
        use_gnn=True
    )

    # Test entity
    entity = {
        'name': 'Dr. Sarah Chen',
        'type': 'Person',
        'description': 'AI Research Director'
    }
    embedding = await resolver._get_embedding(entity['name'] + ' ' + entity['description'])

    # Find candidates
    candidates = await resolver.find_candidates(entity['name'], embedding)

    assert isinstance(candidates, list)
    assert len(candidates) <= 5
    for candidate in candidates:
        assert 'similarity' in candidate
        assert 0 <= candidate['similarity'] <= 1.0
```

---

## 8. SUCCESS CRITERIA

### 8.1 Quantitative Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Entity Resolution F1 | >0.92 | A/B test on test set |
| Link Prediction AUC | >0.88 | ROC AUC on hold-out edges |
| Silhouette Score | >0.45 | Cluster quality metric |
| Training Time | <2 hours | For 10K node graph |
| Query Latency | <100ms | Vector similarity search |

### 8.2 Qualitative Criteria

- [ ] GNN embeddings capture structural patterns (verified by manual inspection)
- [ ] Attention weights interpretable and meaningful
- [ ] Entity resolution improved for edge cases
- [ ] System stable in production (no crashes, reasonable memory)
- [ ] Documentation complete

---

## 9. ROLLBACK PLAN

If GNN embeddings degrade performance:

```python
# Feature flag for easy rollback
USE_GNN_EMBEDDINGS = os.getenv("USE_GNN", "false").lower() == "true"

class EntityResolver:
    async def find_candidates(self, entity_name, embedding):
        if USE_GNN_EMBEDDINGS:
            return await self._find_with_gnn(entity_name, embedding)
        else:
            return await self._find_with_text(entity_name, embedding)
```

**Rollback triggers**:
- Entity resolution F1 decreases by >5%
- System instability (crashes, memory issues)
- User complaints about quality

---

## 10. DOCUMENTATION

### 10.1 Technical Documentation

Create `/docs/gnn_architecture.md`:
- Architecture diagram
- Loss function derivation
- Hyperparameter rationale
- Training procedure

### 10.2 User Documentation

Update `README.md`:
- Explain GNN embeddings
- Benefits over text-only
- How to enable/disable
- Performance characteristics

---

## 11. TIMELINE SUMMARY

| Week | Milestone | Deliverable |
|-------|-----------|--------------|
| 1 | Environment Setup | GPU-ready dev environment |
| 2 | Baseline | GCN baseline + metrics |
| 3 | GAT Model | Heterogeneous GAT implementation |
| 4 | Training Loop | Multi-task training pipeline |
| 5 | Optimization | Hyperparameter search results |
| 6 | Evaluation | ER F1, Link AUC, qualitative |
| 7 | Export | Embeddings in database |
| 8 | Integration | Production-ready entity resolution |

---

**Document Status**: ✅ COMPLETE - READY FOR IMPLEMENTATION
**Estimated Total Effort**: 8 weeks
**Resource Requirements**: GPU recommended (but optional), 16GB RAM minimum
