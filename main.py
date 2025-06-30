from graph_processor import KnowledgeGraphProcessor
from entropy_model import GraphEntropyModel
import numpy as np

def test_pipeline():
    # Sample text
    text = "Alice loves Bob. Bob hates chocolate. Alice admires Eve."
    
    # Build graph
    processor = KnowledgeGraphProcessor()
    kg = processor.build_graph(text)
    
    # Initialize model
    model = GraphEntropyModel()
    
    # Simple test traversal
    start_node = "Alice"
    if start_node in kg.nodes:
        emb = kg.nodes[start_node]['embedding']
        entropy = model(torch.tensor(emb).float())
        print(f"Entropy for {start_node}: {entropy.item():.4f}")

if __name__ == "__main__":
    test_pipeline()