"""
Multimodal MLP Projector for Track 2 (GNN -> LLM Alignment).
Projects GNN topological node embeddings from R^128 into the LLM's token embedding space R^1536
(matching Qwen2.5-1.5B-Instruct hidden dimension).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class GraphToLLMProjector(nn.Module):
    """
    2-Layer Non-Linear Multimodal Projector.
    R^128 (GNN Node Embedding) -> R^512 -> GELU -> LayerNorm -> R^1536 (LLM Token Embedding)
    """
    def __init__(self, gnn_dim: int = 128, hidden_dim: int = 512, llm_dim: int = 1536):
        super().__init__()
        self.fc1 = nn.Linear(gnn_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor of shape [batch_size, num_nodes, 128] or [num_nodes, 128]
        Output: Projected soft token embeddings [batch_size, num_nodes, 1536] or [num_nodes, 1536]
        """
        x = self.fc1(node_embeddings)
        x = self.act(x)
        x = self.norm(x)
        projected = self.fc2(x)
        return projected

def inject_graph_prefix(text_embeddings: torch.Tensor, graph_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Concatenates projected graph node soft tokens as a prefix to the text token embeddings.
    text_embeddings: [batch_size, seq_len, 1536]
    graph_embeddings: [batch_size, num_graph_tokens, 1536]
    returns: [batch_size, num_graph_tokens + seq_len, 1536]
    """
    if graph_embeddings.dim() == 2:
        graph_embeddings = graph_embeddings.unsqueeze(0).expand(text_embeddings.size(0), -1, -1)
    return torch.cat([graph_embeddings, text_embeddings], dim=1)

def test_projector_shapes():
    proj = GraphToLLMProjector(gnn_dim=128, hidden_dim=512, llm_dim=1536)
    dummy_gnn_nodes = torch.randn(4, 128)  # 4 queried entities in a subgraph
    projected = proj(dummy_gnn_nodes)
    print(f"[OK] Projected GNN node shape : {projected.shape} (Expected: [4, 1536])")

    dummy_text_embeds = torch.randn(1, 20, 1536)  # 20 text prompt tokens
    dummy_graph_prefix = projected.unsqueeze(0)    # [1, 4, 1536]
    combined = inject_graph_prefix(dummy_text_embeds, dummy_graph_prefix)
    print(f"[OK] Injected Multimodal shape: {combined.shape} (Expected: [1, 24, 1536])")

if __name__ == "__main__":
    test_projector_shapes()
