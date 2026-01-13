# AUTOMATED KNOWLEDGE GRAPH COMPLETION - IMPLEMENTATION PLAN

**Version**: 1.0
**Date**: January 13, 2026
**Status**: Implementation-Ready
**Priority**: P0
**Estimated Effort**: 9 weeks

---

## 1. EXECUTIVE SUMMARY

**Objective**: Automate discovery of missing entities and relationships using machine learning, reducing manual effort and increasing knowledge coverage.

**Key Deliverables**:
- Link prediction models (TransE + GNN-based)
- Entity discovery models for text processing
- Confidence scoring and human-in-the-loop system
- 42%+ improvement in knowledge coverage
- 26%+ improvement in link precision

**Impact**: Very High - Addresses core completeness problem

---

## 2. TECHNICAL APPROACH

### 2.1 Two-Pronged Strategy

1. **Link Prediction**: Predict missing relationships between existing entities
2. **Entity Completion**: Discover missing entities from new text

**Pipeline Diagram**:
```
New Text ──► Entity Discovery Model
                   │
                   ▼
            Proposed Entities
                   │
                   ├─► LLM Verification
                   │
                   ├─► Human Review
                   │
                   └─► Accept/Reject

Existing Knowledge Graph
                   │
                   ▼
         Link Prediction Model
                   │
                   ▼
         Proposed Links
                   │
                   ├─► Confidence Scoring
                   │
                   ├─► Rule-based Filtering
                   │
                   └─► Auto-accept / Review
```

---

## 3. PHASE 1: LINK PREDICTION (WEEKS 1-5)

### 3.1 Week 1-2: TransE Baseline

**Rationale**: Simple, interpretable baseline for comparison.

**TransE Architecture**:
```python
# knowledge_base/kg_completion/models.py
import torch
import torch.nn as nn
from typing import Dict, List

class TransE(nn.Module):
    """
    Translational embedding model for link prediction.
    Learns: h + r ≈ t
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        Compute score: -||h + r - t||
        Lower distance = better link
        """
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        score = torch.norm(h + r - t, p=2, dim=-1)
        return -score  # Negative for higher = better

    def predict_top_k_links(
        self,
        entity_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """Predict top-k missing links for entity"""

        with torch.no_grad():
            # Get all possible tail combinations
            scores = []

            for rel_id in range(self.num_relations):
                for tail_id in range(self.num_entities):
                    if entity_id == tail_id:
                        continue  # Skip self-links

                    head = torch.tensor([entity_id])
                    relation = torch.tensor([rel_id])
                    tail = torch.tensor([tail_id])

                    score = self(head, relation, tail).item()

                    scores.append({
                        'tail_id': tail_id,
                        'relation_id': rel_id,
                        'score': score
                    })

            # Sort and return top-k
            scores.sort(key=lambda x: x['score'], reverse=True)
            return scores[:top_k]

    def score_triple(self, head: int, relation: int, tail: int) -> float:
        """Score a single triple (h, r, t)"""
        with torch.no_grad():
            h = torch.tensor([head])
            r = torch.tensor([relation])
            t = torch.tensor([tail])
            return self(h, r, t).item()
```

**Training Loop**:
```python
# knowledge_base/kg_completion/training.py
import torch.optim as optim
from tqdm import tqdm

def train_trans_e(
    model: TransE,
    triples: List[Tuple[int, int, int]],
    num_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256,
    margin: float = 1.0
) -> Dict:
    """
    Train TransE with margin-based ranking loss.
    Loss = max(0, ||h+r-t|| + margin - ||h+r-t'||)
    """

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_history = {'loss': []}

    for epoch in tqdm(range(num_epochs), desc="Training TransE"):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle triples
        perm = torch.randperm(len(triples))

        for i in range(0, len(triples), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = [triples[idx] for idx in batch_idx]

            # Positive triples
            heads = torch.tensor([t[0] for t in batch])
            relations = torch.tensor([t[1] for t in batch])
            tails = torch.tensor([t[2] for t in batch])

            # Negative triples (corrupt tail)
            neg_tails = torch.randint(
                0, model.num_entities,
                (len(batch),)
            )
            # Ensure negative tail != positive tail
            neg_tails[neg_tails == tails] = torch.randint(
                0, model.num_entities,
                (neg_tails[neg_tails == tails].shape[0],)
            )

            # Forward pass
            pos_scores = model(heads, relations, tails)
            neg_scores = model(heads, relations, neg_tails)

            # Margin-based ranking loss
            loss = torch.clamp(pos_scores - neg_scores + margin, min=0).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        training_history['loss'].append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return {'model': model, 'history': training_history}
```

**Data Preparation**:
```python
# knowledge_base/kg_completion/data.py
import psycopg
from collections import defaultdict

def load_triples_from_db(db_conn_str: str) -> Tuple[List[Tuple[int,int,int]], Dict, Dict]:
    """
    Load all triples (head, relation, tail) from database.
    Returns:
        - List of triples as (entity_idx, relation_idx, entity_idx)
        - Entity ID to index mapping
        - Relation name to index mapping
    """

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # Get all entities
            cur.execute("SELECT id FROM nodes")
            entity_rows = cur.fetchall()
            entity_ids = [row[0] for row in entity_rows]

            # Create entity mapping
            entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}

            # Get all relations
            cur.execute("SELECT DISTINCT type FROM edges")
            relation_rows = cur.fetchall()
            relation_names = [row[0] for row in relation_rows]

            # Create relation mapping
            relation_to_idx = {name: idx for idx, name in enumerate(relation_names)}

            # Load triples
            cur.execute(
                """
                SELECT source_id, target_id, type
                FROM edges
                """
            )
            edge_rows = cur.fetchall()

            triples = []
            for source_id, target_id, relation_type in edge_rows:
                head_idx = entity_to_idx[source_id]
                tail_idx = entity_to_idx[target_id]
                rel_idx = relation_to_idx[relation_type]

                triples.append((head_idx, rel_idx, tail_idx))

            return triples, entity_to_idx, relation_to_idx

def create_train_test_split(
    triples: List[Tuple[int, int, int]],
    test_ratio: float = 0.2
) -> Tuple[List, List]:
    """Split triples into train and test sets"""

    perm = torch.randperm(len(triples))
    split_idx = int(len(triples) * (1 - test_ratio))

    train_triples = [triples[idx] for idx in perm[:split_idx]]
    test_triples = [triples[idx] for idx in perm[split_idx:]]

    return train_triples, test_triples
```

**Deliverable**: Trained TransE model with baseline metrics

### 3.2 Week 3-4: GNN Link Prediction

**Task**: Implement GNN-based link predictor

```python
# knowledge_base/kg_completion/models.py
from torch_geometric.nn import GCNConv, GATConv

class GNNLinkPredictor(nn.Module):
    """
    GNN-based link prediction.
    Uses GNN to learn node embeddings, then predicts links.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Link prediction head (MLP)
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probability of (head, tail) edge.
        """

        # Get GNN embeddings
        embeddings = x
        for conv in self.convs:
            embeddings = conv(embeddings, edge_index)
            embeddings = F.relu(embeddings)
            embeddings = self.dropout(embeddings)

        # Get head and tail embeddings
        head_emb = embeddings[head]
        tail_emb = embeddings[tail]

        # Concatenate and predict
        combined = torch.cat([head_emb, tail_emb], dim=-1)
        probability = self.link_predictor(combined)

        return probability

    def predict_all_missing_links(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        top_k: int = 100
    ) -> List[Dict]:
        """Predict all missing links"""

        with torch.no_grad():
            # Get GNN embeddings
            embeddings = x
            for conv in self.convs:
                embeddings = conv(embeddings, edge_index)
                embeddings = F.relu(embeddings)

            scores = []
            num_nodes = embeddings.shape[0]

            # Score all pairs (limit to top-k candidates for efficiency)
            for head_id in range(num_nodes):
                for tail_id in range(head_id + 1, num_nodes):
                    head_emb = embeddings[head_id]
                    tail_emb = embeddings[tail_id]

                    # Dot product similarity
                    score = (head_emb * tail_emb).sum().item()

                    scores.append({
                        'head_id': head_id,
                        'tail_id': tail_id,
                        'score': score
                    })

            # Sort and return top-k
            scores.sort(key=lambda x: x['score'], reverse=True)
            return scores[:top_k]
```

**Training Loop**:
```python
def train_gnn_link_predictor(
    model: GNNLinkPredictor,
    train_triples: List[Tuple[int, int, int]],
    num_nodes: int,
    num_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256
) -> Dict:
    """Train GNN link predictor"""

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create edge index from train triples
    edge_index = torch.tensor([
        [t[0] for t in train_triples],
        [t[2] for t in train_triples]
    ], dtype=torch.long)

    training_history = {'loss': []}

    for epoch in tqdm(range(num_epochs), desc="Training GNN Link Predictor"):
        epoch_loss = 0.0
        num_batches = 0

        perm = torch.randperm(len(train_triples))

        for i in range(0, len(train_triples), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = [train_triples[idx] for idx in batch_idx]

            heads = torch.tensor([t[0] for t in batch])
            relations = torch.tensor([t[1] for t in batch])
            tails = torch.tensor([t[2] for t in batch])

            # Forward pass
            probs = model(edge_index, heads, tails)

            # Binary cross-entropy loss
            targets = torch.ones(probs.shape[0])  # All training triples are positive
            loss = F.binary_cross_entropy(probs.squeeze(), targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        training_history['loss'].append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return {'model': model, 'history': training_history}
```

**Deliverable**: Trained GNN link predictor

### 3.3 Week 5: Ensemble & Comparison

**Task**: Combine TransE and GNN predictions, evaluate

```python
# knowledge_base/kg_completion/ensemble.py
from typing import List

class EnsembleLinkPredictor:
    """
    Ensemble of multiple link prediction models.
    Combines predictions with weighted voting.
    """

    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, head: int, tail: int) -> Dict:
        """Combine predictions from all models"""

        predictions = []

        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'score_triple'):
                # TransE
                score = model.score_triple(head, 0, tail)  # Default relation
                predictions.append({'model': 'TransE', 'score': score, 'weight': weight})
            else:
                # GNN
                score = model(edge_index, head, tail).item()
                predictions.append({'model': 'GNN', 'score': score, 'weight': weight})

        # Weighted average
        ensemble_score = sum(
            p['score'] * p['weight'] for p in predictions
        )

        return {
            'ensemble_score': ensemble_score,
            'individual_scores': predictions
        }
```

**Evaluation**:
```python
# knowledge_base/kg_completion/evaluation.py
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_link_prediction(
    model: nn.Module,
    test_triples: List[Tuple[int, int, int]],
    neg_triples: List[Tuple[int, int, int]]
) -> Dict:
    """Evaluate link prediction performance"""

    model.eval()

    # Predict scores
    pos_scores = []
    neg_scores = []

    with torch.no_grad():
        for head, relation, tail in test_triples:
            if hasattr(model, 'score_triple'):
                # TransE
                score = model.score_triple(head, relation, tail)
            else:
                # GNN
                score = model(edge_index, head, tail).item()
            pos_scores.append(score)

        for head, relation, tail in neg_triples:
            if hasattr(model, 'score_triple'):
                score = model.score_triple(head, relation, tail)
            else:
                score = model(edge_index, head, tail).item()
            neg_scores.append(score)

    # Compute metrics
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores

    auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    return {
        'auc': auc,
        'average_precision': avg_precision
    }
```

**Deliverable**: Ensemble model with performance comparison

---

## 4. PHASE 2: ENTITY COMPLETION (WEEKS 6-7)

### 4.1 Week 6: Entity Discovery Model

**Task**: Train model to detect entities in text (NER)

```python
# knowledge_base/kg_completion/entity_models.py
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer

class EntityDiscoveryModel:
    """
    Fine-tune BERT for named entity recognition.
    Detects entities: Person, Organization, Project, Concept
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 5  # O + PER + ORG + PRO + CON
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        self.label_map = {
            0: 'O',
            1: 'PER',
            2: 'ORG',
            3: 'PRO',
            4: 'CON'
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def prepare_training_data(
        self,
        annotated_texts: List[Dict[str, List]]
    ) -> Dict:
        """
        Prepare training data for BERT fine-tuning.
        Format: {"tokens": [...], "labels": [...]}
        """

        examples = []

        for text in annotated_texts:
            tokens = text['tokens']
            labels = [self.reverse_label_map[l] for l in text['labels']]

            # Tokenize
            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_offsets_mapping=True
            )

            # Align labels with tokens
            label_ids = []
            last_word_idx = -1
            for offset in encoded['offset_mapping']:
                if offset[0] == 0:  # Special token
                    label_ids.append(-100)
                elif offset[0] == offset[1]:  # Subword token
                    label_ids.append(label_ids[-1] if label_ids else -100)
                else:
                    last_word_idx += 1
                    label_ids.append(labels[last_word_idx])

            examples.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': label_ids
            })

        return examples

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text"""

        self.model.eval()

        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**encoded)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Extract entity spans
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets = encoded['offset_mapping'][0]

        current_entity = None

        for i, (token, pred, offset) in enumerate(zip(tokens, predictions, offsets)):
            label = self.label_map[pred.item()]

            if label != 'O':
                if current_entity is None:
                    current_entity = {
                        'type': label,
                        'text': '',
                        'start': offset[0]
                    }
                current_entity['text'] += token.replace('##', '')
            else:
                if current_entity:
                    current_entity['end'] = offset[0]
                    entities.append(current_entity)
                    current_entity = None

        return entities
```

**Training Data Preparation**:
```python
def create_training_data_from_kb(db_conn_str: str) -> List[Dict]:
    """
    Create NER training data from existing KB entities.
    """

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # Get entities with descriptions
            cur.execute(
                """
                SELECT name, type, description
                FROM nodes
                """
            )
            rows = cur.fetchall()

    # Create training examples (simplified)
    training_data = []
    for name, entity_type, description in rows:
        # Create simple annotated text
        text = f"{name} is {description.lower()}"

        # Map entity type to label
        label_map = {
            'Person': 'PER',
            'Organization': 'ORG',
            'Project': 'PRO',
            'Concept': 'CON'
        }
        label = label_map.get(entity_type, 'ORG')

        # Tokenize and label
        tokens = text.split()
        labels = ['O'] * len(tokens)

        # Find entity position and label
        name_tokens = name.split()
        for i in range(len(tokens) - len(name_tokens) + 1):
            if tokens[i:i+len(name_tokens)] == name_tokens:
                for j in range(len(name_tokens)):
                    labels[i + j] = label
                break

        training_data.append({
            'tokens': tokens,
            'labels': labels
        })

    return training_data
```

**Deliverable**: Trained entity discovery model

### 4.2 Week 7: Entity Typing & Integration

**Task**: Classify entity types and integrate with pipeline

```python
class EntityTyper:
    """
    Classify entity types based on context.
    """

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4  # PER, ORG, PRO, CON
        )

        self.label_map = {
            0: 'Person',
            1: 'Organization',
            2: 'Project',
            3: 'Concept'
        }

    def classify(self, entity_text: str, context: str) -> Dict:
        """Classify entity type"""

        input_text = f"Entity: {entity_text}\nContext: {context}"

        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = F.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs).item()

        return {
            'type': self.label_map[prediction],
            'confidence': probs[0][prediction].item()
        }
```

**Deliverable**: Integrated entity discovery and typing

---

## 5. PHASE 3: CONFIDENCE SCORING (WEEK 8)

### 5.1 Week 8: Multi-Factor Confidence Scoring

**Task**: Implement confidence scoring combining model output, structural features, and rules.

```python
# knowledge_base/kg_completion/confidence.py
from typing import Dict, List

class LinkConfidenceScorer:
    """
    Score confidence of predicted links using multiple factors.
    """

    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.70
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def score_link(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
        model_score: float,
        graph: nx.Graph,
        entity_info: Dict
    ) -> Dict:
        """
        Compute confidence score for predicted link.
        Combines: model score, structural features, semantic features.
        """

        # 1. Model confidence (from ML model)
        model_confidence = model_score

        # 2. Structural features
        structural_score = self._compute_structural_score(
            head_id, tail_id, graph
        )

        # 3. Semantic features
        semantic_score = self._compute_semantic_score(
            head_id, tail_id, relation_type, entity_info
        )

        # 4. Rule-based features
        rule_score = self._compute_rule_score(
            head_id, tail_id, relation_type, graph
        )

        # Weighted combination
        weights = {
            'model': 0.5,
            'structural': 0.25,
            'semantic': 0.15,
            'rule': 0.10
        }

        final_confidence = (
            weights['model'] * model_confidence +
            weights['structural'] * structural_score +
            weights['semantic'] * semantic_score +
            weights['rule'] * rule_score
        )

        # Decision
        if final_confidence >= self.high_threshold:
            decision = 'AUTO_ACCEPT'
        elif final_confidence >= self.medium_threshold:
            decision = 'REVIEW'
        else:
            decision = 'REJECT'

        return {
            'confidence': final_confidence,
            'decision': decision,
            'breakdown': {
                'model': model_confidence,
                'structural': structural_score,
                'semantic': semantic_score,
                'rule': rule_score
            }
        }

    def _compute_structural_score(
        self, head_id: str, tail_id: str, graph: nx.Graph
    ) -> float:
        """Compute structural similarity score"""

        score = 0.0

        # Mutual neighbors (triadic closure)
        head_neighbors = set(graph.neighbors(head_id))
        tail_neighbors = set(graph.neighbors(tail_id))
        mutual_neighbors = len(head_neighbors & tail_neighbors)

        score += 0.3 * (min(mutual_neighbors, 5) / 5.0)

        # Path length (shorter = more likely)
        try:
            path_length = nx.shortest_path_length(graph, head_id, tail_id)
            if path_length == 2:  # 2-hop connection
                score += 0.4
            elif path_length == 3:
                score += 0.2
        except nx.NetworkXNoPath:
            score += 0.0

        # Degree centrality (popular nodes more likely to connect)
        head_degree = graph.degree(head_id)
        tail_degree = graph.degree(tail_id)
        avg_degree = (head_degree + tail_degree) / 2.0

        score += 0.3 * min(avg_degree / 10.0, 1.0)

        return min(score, 1.0)

    def _compute_semantic_score(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
        entity_info: Dict
    ) -> float:
        """Compute semantic similarity score"""

        head_info = entity_info.get(head_id, {})
        tail_info = entity_info.get(tail_id, {})

        score = 0.0

        # Type compatibility
        if head_info.get('type') == tail_info.get('type'):
            score += 0.3

        # Description similarity (using embeddings)
        if 'embedding' in head_info and 'embedding' in tail_info:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            head_emb = np.array(head_info['embedding']).reshape(1, -1)
            tail_emb = np.array(tail_info['embedding']).reshape(1, -1)

            sim = cosine_similarity(head_emb, tail_emb)[0][0]
            score += 0.7 * sim

        return min(score, 1.0)

    def _compute_rule_score(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
        graph: nx.Graph
    ) -> float:
        """Compute rule-based score"""

        score = 0.0

        # Rule 1: Duplicate relation check
        if graph.has_edge(head_id, tail_id):
            score -= 1.0  # Strong penalty

        # Rule 2: Entity type constraints
        head_type = entity_info.get(head_id, {}).get('type')
        tail_type = entity_info.get(tail_id, {}).get('type')

        # Valid relations by type
        valid_relations = {
            ('Person', 'Organization'): ['WORKS_FOR', 'LEADS'],
            ('Person', 'Project'): ['AUTHORED', 'WORKS_ON'],
            ('Organization', 'Project'): ['FUNDS', 'OWNS']
        }

        if (head_type, tail_type) in valid_relations:
            if relation_type in valid_relations[(head_type, tail_type)]:
                score += 0.5
            else:
                score -= 0.3  # Unusual relation

        return max(score, 0.0)  # Clip to 0
```

**Deliverable**: Working confidence scoring system

---

## 6. PHASE 4: HUMAN-IN-THE-LOOP (WEEK 9)

### 6.1 Week 9: Review Interface & Integration

**Task**: Build interface for reviewing predicted links/entities.

```python
# knowledge_base/kg_completion/review.py
from pydantic import BaseModel

class ProposedLink(BaseModel):
    head_id: str
    tail_id: str
    relation_type: str
    confidence: float
    decision: str  # AUTO_ACCEPT, REVIEW, REJECT
    breakdown: Dict

class ReviewQueue:
    """
    Manage review queue for predicted links/entities.
    """

    def __init__(self, db_conn_str: str):
        self.db_conn_str = db_conn_str
        self.queue: List[ProposedLink] = []

    def add_proposed_links(self, links: List[ProposedLink]):
        """Add links to review queue"""

        for link in links:
            if link['decision'] == 'REVIEW':
                self.queue.append(link)

    def get_pending_reviews(self, limit: int = 10) -> List[ProposedLink]:
        """Get pending reviews"""

        return self.queue[:limit]

    def approve_link(self, link_id: str, reviewer: str):
        """Approve and add link to KB"""

        link = self._find_link(link_id)

        if link:
            # Add to database
            self._insert_link(link)
            self._remove_from_queue(link_id)

    def reject_link(self, link_id: str, reviewer: str, reason: str):
        """Reject and log reason"""

        link = self._find_link(link_id)

        if link:
            # Log rejection
            self._log_rejection(link, reviewer, reason)
            self._remove_from_queue(link_id)

    def auto_process_high_confidence(self):
        """Auto-accept high-confidence links"""

        high_conf = [l for l in self.queue if l['decision'] == 'AUTO_ACCEPT']

        for link in high_conf:
            self._insert_link(link)
            self._remove_from_queue(link['head_id'], link['tail_id'])

        return len(high_conf)

    def _insert_link(self, link: ProposedLink):
        """Insert approved link into database"""

        with psycopg.connect(self.db_conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO edges (source_id, target_id, type, description, weight)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        link.head_id,
                        link.tail_id,
                        link.relation_type,
                        f"Auto-generated by KG completion (confidence: {link.confidence:.2f})",
                        link.confidence
                    )
                )
                conn.commit()
```

**Deliverable**: Complete review system integrated

---

## 7. EVALUATION

### 7.1 Link Prediction Metrics

```python
def comprehensive_link_evaluation(
    model: nn.Module,
    test_triples: List,
    neg_triples: List,
    k_values: List[int] = [10, 50, 100]
) -> Dict:
    """Comprehensive evaluation of link prediction"""

    model.eval()

    metrics = {}

    # AUC
    metrics['auc'] = evaluate_link_prediction_auc(
        model, test_triples, neg_triples
    )

    # Precision@K
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(
            model, test_triples, neg_triples, k
        )

    # Recall@K
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(
            model, test_triples, neg_triples, k
        )

    # MRR (Mean Reciprocal Rank)
    metrics['mrr'] = mean_reciprocal_rank(
        model, test_triples, neg_triples
    )

    return metrics
```

### 7.2 Entity Discovery Metrics

```python
def evaluate_entity_discovery(
    model: EntityDiscoveryModel,
    test_documents: List[Dict]
) -> Dict:
    """Evaluate entity discovery performance"""

    total_entities = 0
    detected_entities = 0
    correct_types = 0

    for doc in test_documents:
        ground_truth = doc['entities']  # List of (text, type)
        predictions = model.extract_entities(doc['text'])

        # Detection metrics
        predicted_texts = [p['text'] for p in predictions]
        for entity_text, entity_type in ground_truth:
            total_entities += 1
            if entity_text in predicted_texts:
                detected_entities += 1

                # Type accuracy
                pred = [p for p in predictions if p['text'] == entity_text][0]
                if pred['type'] == entity_type:
                    correct_types += 1

    recall = detected_entities / total_entities if total_entities > 0 else 0
    type_accuracy = correct_types / detected_entities if detected_entities > 0 else 0

    return {
        'recall': recall,
        'type_accuracy': type_accuracy
    }
```

---

## 8. SUCCESS CRITERIA

### 8.1 Quantitative Metrics

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Link Precision@100 | 0.65 | 0.82 | +26% |
| Link Recall@100 | 0.58 | 0.75 | +29% |
| Entity Detection Recall | 0.72 | 0.88 | +22% |
| Knowledge Coverage | 60% | 85% | +42% |
| Auto-Accept Rate | N/A | 70% | N/A |

### 8.2 Qualitative Criteria

- [ ] Human reviewers approve >80% of auto-accepted links
- [ ] Confidence scores correlate with human judgment
- [ ] System discovers meaningful links (validated by domain experts)
- [ ] No obvious false positives in top-100 predictions
- [ ] Review queue manageable (<50 pending items/day)

---

## 9. ROLLBACK PLAN

**Triggers**:
- Link precision decreases by >10%
- Entity type accuracy <70%
- Human rejection rate >50%
- System performance degradation (latency >2s)

**Rollback Actions**:
```python
# Feature flag
ENABLE_KG_COMPLETION = os.getenv("ENABLE_KG_COMPLETION", "false").lower() == "true"

class Pipeline:
    def run(self, file_path):
        graph = await self.ingestor.extract(text)

        # Standard pipeline
        await self._store_graph(graph)

        # KG completion (if enabled)
        if ENABLE_KG_COMPLETION:
            await self._complete_knowledge_graph()
```

---

## 10. TIMELINE SUMMARY

| Week | Milestone | Deliverable |
|-------|-----------|--------------|
| 1-2 | TransE Baseline | Trained TransE model |
| 3-4 | GNN Link Predictor | Trained GNN model |
| 5 | Ensemble | Combined predictor |
| 6 | Entity Discovery | NER model |
| 7 | Entity Typing | Classifier + integration |
| 8 | Confidence Scoring | Multi-factor scorer |
| 9 | Review System | Human-in-the-loop |

---

**Document Status**: ✅ COMPLETE - READY FOR IMPLEMENTATION
**Estimated Total Effort**: 9 weeks
**Resource Requirements**: GPU recommended, 16GB RAM minimum
