"""
Relational Graph Convolutional Network (RGCN) Encoder & Link Predictor for Track 2.
Encodes multi-relational knowledge graph topology into node embeddings (d=128),
and performs relational link prediction across the 5 ontology relations.
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class RelationalGNNEncoder(nn.Module):
    """
    2-Layer Relational Graph Convolutional Network (RGCN).
    Propagates structural messages across relation-specific adjacency matrices.
    """
    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 128, num_relations: int = 5, num_bases: int = 5):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_type)
        return h

class RelationalLinkPredictor(nn.Module):
    """
    DistMult Bilinear Scoring Function for Multi-Relational Link Prediction.
    score(u, r, v) = sum(h_u * W_r * h_v)
    """
    def __init__(self, num_relations: int = 5, embed_dim: int = 128):
        super().__init__()
        # Diagonal relation embedding matrix
        self.rel_embed = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def score(self, h_src: torch.Tensor, h_dst: torch.Tensor, r_id: torch.Tensor) -> torch.Tensor:
        r_w = self.rel_embed(r_id)
        # Element-wise product then sum over embedding dimension
        return torch.sum(h_src * r_w * h_dst, dim=-1)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]
        logits = self.score(h_src, h_dst, edge_type)
        return logits

    def predict_relation(self, h: torch.Tensor, u_idx: int, v_idx: int, num_relations: int = 5) -> Tuple[int, torch.Tensor]:
        """
        Given head node u and tail node v, evaluates all relations r in [0..num_relations-1]
        and returns the argmax relation ID with prediction probabilities.
        """
        h_u = h[u_idx].unsqueeze(0).repeat(num_relations, 1)
        h_v = h[v_idx].unsqueeze(0).repeat(num_relations, 1)
        all_r = torch.arange(num_relations, device=h.device)
        scores = self.score(h_u, h_v, all_r)
        probs = torch.softmax(scores, dim=-1)
        best_r = torch.argmax(probs).item()
        return best_r, probs

class HybridGNNLinkModel(nn.Module):
    """
    Unified GNN Encoder + Link Predictor Model.
    """
    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 128, num_relations: int = 5):
        super().__init__()
        self.encoder = RelationalGNNEncoder(in_dim, hidden_dim, out_dim, num_relations)
        self.predictor = RelationalLinkPredictor(num_relations, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, query_edges: torch.Tensor, query_types: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index, edge_type)
        logits = self.predictor(h, query_edges, query_types)
        return logits

def train_gnn_link_prediction(splits: Dict[str, Any], epochs: int = 120, lr: float = 0.01) -> Tuple[HybridGNNLinkModel, Dict[str, float]]:
    """
    Trains the Relational GNN on Knowledge Graph edges with Binary Cross-Entropy loss.
    """
    num_nodes = splits["num_nodes"]
    x = splits["x"]
    train_data = splits["train"]
    val_data = splits["val"]
    test_data = splits["test"]

    model = HybridGNNLinkModel(in_dim=x.size(1), hidden_dim=128, out_dim=128, num_relations=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    print("\n🚀 Training Relational GNN on Knowledge Graph Adjacency...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Full message passing over positive training topology
        h = model.encoder(x, train_data["pos_edge_index"], train_data["pos_edge_type"])

        # Score positive and negative training edges
        pos_logits = model.predictor(h, train_data["pos_edge_index"], train_data["pos_edge_type"])
        neg_logits = model.predictor(h, train_data["neg_edge_index"], train_data["neg_edge_type"])

        all_logits = torch.cat([pos_logits, neg_logits])
        all_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])

        loss = criterion(all_logits, all_labels)
        loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                h_val = model.encoder(x, train_data["pos_edge_index"], train_data["pos_edge_type"])
                val_pos = model.predictor(h_val, val_data["pos_edge_index"], val_data["pos_edge_type"])
                val_neg = model.predictor(h_val, val_data["neg_edge_index"], val_data["neg_edge_type"])

                val_pos_acc = (torch.sigmoid(val_pos) > 0.5).float().mean().item()
                val_neg_acc = (torch.sigmoid(val_neg) <= 0.5).float().mean().item()
                val_acc = (val_pos_acc + val_neg_acc) / 2.0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = model.state_dict()

                if epoch % 20 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Link Acc: {val_acc*100:.2f}% (Pos: {val_pos_acc*100:.1f}%, Neg: {val_neg_acc*100:.1f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test Set Evaluation
    model.eval()
    with torch.no_grad():
        h_test = model.encoder(x, train_data["pos_edge_index"], train_data["pos_edge_type"])
        test_pos = model.predictor(h_test, test_data["pos_edge_index"], test_data["pos_edge_type"])
        test_neg = model.predictor(h_test, test_data["neg_edge_index"], test_data["neg_edge_type"])
        test_pos_acc = (torch.sigmoid(test_pos) > 0.5).float().mean().item()
        test_neg_acc = (torch.sigmoid(test_neg) <= 0.5).float().mean().item()
        test_acc = (test_pos_acc + test_neg_acc) / 2.0

    metrics = {
        "final_train_loss": loss.item(),
        "best_val_accuracy": best_val_acc,
        "test_link_accuracy": test_acc
    }
    return model, metrics

def main():
    from llm_graph_ontology.src.data.graph_dataset import (
        load_graph_entities_and_triples,
        build_pyg_graph,
        split_graph_edges
    )
    node_to_id, triples, _ = load_graph_entities_and_triples()
    data = build_pyg_graph(node_to_id, triples)
    splits = split_graph_edges(data)

    model, metrics = train_gnn_link_prediction(splits, epochs=100)

    print("\n" + "=" * 60)
    print("🏆 GNN STANDALONE TOPOLOGICAL LINK PREDICTION RESULTS")
    print("=" * 60)
    print(f"Test Link Accuracy (Topological GNN) : {metrics['test_link_accuracy'] * 100:.2f}%")
    print(f"Validation Accuracy                   : {metrics['best_val_accuracy'] * 100:.2f}%")
    print("=" * 60)

    out_dir = Path("llm-graph-ontology/models/gnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "rgcn_legal_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Saved trained GNN weights to: {model_path}")

if __name__ == "__main__":
    # Support direct execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data.graph_dataset import (
        load_graph_entities_and_triples,
        build_pyg_graph,
        split_graph_edges
    )
    node_to_id, triples, _ = load_graph_entities_and_triples()
    data = build_pyg_graph(node_to_id, triples)
    splits = split_graph_edges(data)

    model, metrics = train_gnn_link_prediction(splits, epochs=100)

    print("\n" + "=" * 60)
    print("🏆 GNN STANDALONE TOPOLOGICAL LINK PREDICTION RESULTS")
    print("=" * 60)
    print(f"Test Link Accuracy (Topological GNN) : {metrics['test_link_accuracy'] * 100:.2f}%")
    print(f"Validation Accuracy                   : {metrics['best_val_accuracy'] * 100:.2f}%")
    print("=" * 60)

    out_dir = Path("llm-graph-ontology/models/gnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "rgcn_legal_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Saved trained GNN weights to: {model_path}")
