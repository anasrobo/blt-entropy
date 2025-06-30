import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import deque

# Simplified demo script for entropy-guided traversal
# Uses random embeddings and splits on periods to avoid external dependencies

class GraphEntropyModel(nn.Module):
    """Simple transformer-based entropy prediction head"""
    def __init__(self, embedding_dim=16, hidden_dim=32, num_layers=1):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        self.entropy_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (seq_len, batch=1, embed_dim)
        out = self.transformer(x)
        return self.entropy_head(out[-1])

class SimpleGraphProcessor:
    """Builds a graph from demo text and assigns random embeddings"""
    def __init__(self, embedding_dim=16):
        self.graph = nx.DiGraph()
        self.embedding_dim = embedding_dim

    def build_graph(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        print(f"Building graph from sentences: {sentences}")
        self.graph.clear()
        for idx, sent in enumerate(sentences):
            words = [w.lower() for w in sent.split()]
            if len(words) >= 2:
                subj, obj = words[0], words[-1]
                emb_s = np.random.rand(self.embedding_dim)
                emb_o = np.random.rand(self.embedding_dim)
                self.graph.add_node(subj, embedding=emb_s, sentence=idx)
                self.graph.add_node(obj, embedding=emb_o, sentence=idx)
                self.graph.add_edge(subj, obj)
                self.graph.add_edge(obj, subj)
        print(f"Graph nodes: {list(self.graph.nodes)}")
        return self.graph

    def entropy_guided_traversal(self, start, model, threshold=0.5, max_depth=5):
        visited = {start}
        scores = {}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            seq_embs = [self.graph.nodes[n]['embedding'] for n in path]
            tensor = torch.tensor(seq_embs, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                score = model(tensor).item()
            scores[node] = score
            if score > threshold or len(path) > max_depth:
                continue
            for nbr in self.graph.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
        return scores

if __name__ == '__main__':
    # Define multiple demo texts to test
    demo_texts = [
        "Alice loves Bob. Bob trusts Charlie. Charlie helps Alice",
        "Paris is beautiful. The Eiffel tower stands tall. Tourists admire Paris",
        "Data drives insights. Insights fuel decisions. Decisions shape futures"
    ]

    processor = SimpleGraphProcessor(embedding_dim=16)
    model = GraphEntropyModel(embedding_dim=16, hidden_dim=32)
    # No training; using random model weights for demonstration

    for i, text in enumerate(demo_texts, 1):
        print(f"\n=== Test #{i} ===")
        graph = processor.build_graph(text)
        start_node = list(graph.nodes)[0]  # pick the first node as start
        print(f"Starting traversal from node: '{start_node}'")
        scores = processor.entropy_guided_traversal(start_node, model)
        print("Entropy scores:")
        for node, sc in scores.items():
            print(f" {node}: {sc:.4f}")
