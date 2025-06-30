import torch
import torch.nn as nn

class GraphEntropyModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings):
        encoded = self.encoder(node_embeddings)
        return self.entropy_head(encoded.mean(dim=0))