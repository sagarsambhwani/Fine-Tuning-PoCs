"""
Graph Dataset & Topology Builder for Track 2 (PyTorch Geometric).
Converts the Corporate Litigation Knowledge Graph into a PyG Data object
with multi-relational edges and negative edge sampling for Link Prediction.
"""
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from torch_geometric.data import Data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 5 Closed-World Ontology Relations
RELATION_MAP = {
    "FILES_CLAIM_AGAINST": 0,
    "ADJUDICATED_BY": 1,
    "APPLIES_PRECEDENT": 2,
    "ESTABLISHES_REMEDY": 3,
    "LIABLE_FOR": 4
}
INV_RELATION_MAP = {v: k for k, v in RELATION_MAP.items()}

# Entity Categories for Initial Type Features
ENTITY_TYPES = ["LegalParty", "Court", "LegalClaim", "PrecedentCase", "LegalRemedy"]
TYPE_TO_ID = {t: i for i, t in enumerate(ENTITY_TYPES)}

def load_graph_entities_and_triples(train_file_path: str = None) -> Tuple[Dict[str, int], List[Tuple[int, int, int]], Dict[int, int]]:
    """
    Extracts all unique entities, node mappings, entity types, and graph edges (triplets)
    from the processed training data and schema.
    """
    if train_file_path is None:
        candidates = [
            "llm-graph-ontology/data/processed/train.jsonl",
            "data/processed/train.jsonl",
            "../data/processed/train.jsonl"
        ]
        for c in candidates:
            if Path(c).exists():
                train_file_path = c
                break
        if train_file_path is None:
            raise FileNotFoundError("Could not locate train.jsonl for graph construction.")

    node_to_id = {}
    id_to_type = {}
    triples = set()

    with open(train_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Parse from triplet extraction or multihop reasoning
            if item.get("task_type") == "triplet_extraction":
                try:
                    assistant_content = item["messages"][2]["content"]
                    parsed = json.loads(assistant_content)
                    if isinstance(parsed, list):
                        for t in parsed:
                            sub = t.get("subject", "").strip()
                            pred = t.get("predicate", "").strip()
                            obj = t.get("object", "").strip()

                            if sub and pred in RELATION_MAP and obj:
                                if sub not in node_to_id:
                                    node_to_id[sub] = len(node_to_id)
                                if obj not in node_to_id:
                                    node_to_id[obj] = len(node_to_id)

                                s_id = node_to_id[sub]
                                o_id = node_to_id[obj]
                                r_id = RELATION_MAP[pred]
                                triples.add((s_id, r_id, o_id))
                except Exception:
                    pass

    return node_to_id, list(triples), id_to_type

def build_pyg_graph(node_to_id: Dict[str, int], triples: List[Tuple[int, int, int]], in_dim: int = 64, seed: int = 42) -> Data:
    """
    Builds a PyTorch Geometric Data object with:
    - x: Initial node features [num_nodes, in_dim]
    - edge_index: [2, num_edges]
    - edge_type: [num_edges]
    """
    torch.manual_seed(seed)
    num_nodes = len(node_to_id)

    # Initial node representations: normalized randomized orthogonal embeddings
    # Provides unique topological identifiers for each entity node
    x = torch.randn(num_nodes, in_dim)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)

    edge_list_src = []
    edge_list_dst = []
    edge_types = []

    for s, r, o in triples:
        edge_list_src.append(s)
        edge_list_dst.append(o)
        edge_types.append(r)

    edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = num_nodes
    data.num_relations = len(RELATION_MAP)
    return data

def sample_negative_edges(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples negative edges (u, r, v') where no edge exists between u and v' for relation r.
    Returns neg_edge_index [2, num_edges] and neg_edge_type [num_edges].
    """
    rng = random.Random(seed)
    existing_edges = set((edge_index[0, i].item(), edge_type[i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1)))

    neg_src = []
    neg_dst = []
    neg_rel = []

    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        r = edge_type[i].item()
        # Corrupt tail entity
        for _ in range(100):
            corrupt_v = rng.randint(0, num_nodes - 1)
            if (u, r, corrupt_v) not in existing_edges and corrupt_v != u:
                neg_src.append(u)
                neg_dst.append(corrupt_v)
                neg_rel.append(r)
                break
        else:
            # Fallback
            neg_src.append(u)
            neg_dst.append((u + 1) % num_nodes)
            neg_rel.append(r)

    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long)
    neg_edge_type = torch.tensor(neg_rel, dtype=torch.long)
    return neg_edge_index, neg_edge_type

def split_graph_edges(data: Data, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Dict[str, Any]:
    """
    Splits edges into train, validation, and test sets with matching negative edge samples.
    """
    rng = random.Random(seed)
    num_edges = data.edge_index.size(1)
    indices = list(range(num_edges))
    rng.shuffle(indices)

    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)
    num_train = num_edges - num_val - num_test

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train + num_val]
    test_idx = indices[num_train + num_val:]

    def get_subset(idx_list):
        pos_ei = data.edge_index[:, idx_list]
        pos_et = data.edge_type[idx_list]
        neg_ei, neg_et = sample_negative_edges(pos_ei, pos_et, data.num_nodes, seed=seed + len(idx_list))
        return {
            "pos_edge_index": pos_ei,
            "pos_edge_type": pos_et,
            "neg_edge_index": neg_ei,
            "neg_edge_type": neg_et
        }

    return {
        "train": get_subset(train_idx),
        "val": get_subset(val_idx),
        "test": get_subset(test_idx),
        "num_nodes": data.num_nodes,
        "x": data.x
    }

def main():
    node_to_id, triples, _ = load_graph_entities_and_triples()
    data = build_pyg_graph(node_to_id, triples)
    splits = split_graph_edges(data)

    print("=" * 60)
    print("📊 KNOWLEDGE GRAPH TOPOLOGY AUDIT (PyTorch Geometric)")
    print("=" * 60)
    print(f"Total Unique Entity Nodes : {data.num_nodes}")
    print(f"Total Relational Edges    : {data.edge_index.size(1)}")
    print(f"Ontology Relations Count  : {data.num_relations}")
    print(f"Train Edges (Pos / Neg)   : {splits['train']['pos_edge_index'].size(1)} / {splits['train']['neg_edge_index'].size(1)}")
    print(f"Val Edges (Pos / Neg)     : {splits['val']['pos_edge_index'].size(1)} / {splits['val']['neg_edge_index'].size(1)}")
    print(f"Test Edges (Pos / Neg)    : {splits['test']['pos_edge_index'].size(1)} / {splits['test']['neg_edge_index'].size(1)}")
    print("=" * 60)

    # Save mapping for inference lookup
    out_dir = Path("llm-graph-ontology/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = out_dir / "graph_node_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({"node_to_id": node_to_id, "relation_map": RELATION_MAP}, f, indent=2)
    print(f"[SUCCESS] Node mapping saved to: {mapping_file}")

if __name__ == "__main__":
    main()
