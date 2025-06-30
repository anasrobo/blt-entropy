import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
from collections import deque, defaultdict

# Download required NLTK data
nltk.download('punkt')

class GraphEntropyModel(nn.Module):
    """Adapted BLT model for graph entropy calculation"""
    def __init__(self, embedding_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Transformer encoder for sequence processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        
        # Entropy prediction head
        self.entropy_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings):
        # Process node sequence
        transformer_out = self.transformer(node_embeddings)
        
        # Use last output for entropy prediction
        entropy_score = self.entropy_head(transformer_out[-1])
        return entropy_score

class KnowledgeGraphProcessor:
    """Process text into knowledge graph and handle traversal"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.sentence_boundaries = {}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = BertModel.from_pretrained('bert-base-uncased')
        
    def get_node_embedding(self, text):
        """Get BERT embedding for node text"""
        if text not in self.node_embeddings:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = self.embedder(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            self.node_embeddings[text] = embedding
        return self.node_embeddings[text]
    
    def extract_triplets(self, sentence, sent_idx):
        """Improved triplet extraction using dependency parsing heuristics"""
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        
        # Find verbs as potential predicates
        verbs = [i for i, (word, pos) in enumerate(tagged) 
                if pos.startswith('VB')]
        
        triplets = []
        for verb_idx in verbs:
            # Find subject (left of verb)
            subj = " ".join([word for word, pos in tagged[:verb_idx] 
                            if pos in ['NN', 'NNS', 'NNP', 'PRP']])
            
            # Find object (right of verb)
            obj = " ".join([word for word, pos in tagged[verb_idx+1:] 
                           if pos in ['NN', 'NNS', 'NNP', 'PRP']])
            
            if subj and obj:
                triplets.append((subj.lower(), words[verb_idx].lower(), obj.lower()))
                # Record sentence boundary information
                self.sentence_boundaries[subj.lower()] = sent_idx
                self.sentence_boundaries[obj.lower()] = sent_idx
        
        return triplets

    def build_graph(self, text):
        """Construct knowledge graph from text"""
        sentences = sent_tokenize(text)
        for sent_idx, sent in enumerate(sentences):
            triplets = self.extract_triplets(sent, sent_idx)
            for s, v, o in triplets:
                # Get or create embeddings
                s_emb = self.get_node_embedding(s)
                o_emb = self.get_node_embedding(o)
                
                # Add nodes and edges
                self.graph.add_node(s, embedding=s_emb, sentence=sent_idx)
                self.graph.add_node(o, embedding=o_emb, sentence=sent_idx)
                self.graph.add_edge(s, o, predicate=v, sentence=sent_idx)
                
                # Add reverse edge for traversal
                self.graph.add_edge(o, s, predicate=f"rev_{v}", sentence=sent_idx)
        
        return self.graph

    def entropy_guided_traversal(self, start_node, entropy_model, threshold=0.5, max_depth=20):
        """Entropy-guided BFS traversal with boundary detection"""
        # Initialize tracking structures
        visited = set()
        entropy_scores = {}
        queue = deque([(start_node, [start_node])])
        visited.add(start_node)
        
        while queue:
            current, path = queue.popleft()
            current_emb = self.graph.nodes[current]['embedding']
            
            # Prepare sequence of node embeddings
            sequence_embs = np.array([self.graph.nodes[n]['embedding'] for n in path])
            sequence_tensor = torch.tensor(sequence_embs).float().unsqueeze(1)
            
            # Predict entropy score
            with torch.no_grad():
                entropy_score = entropy_model(sequence_tensor).item()
            
            entropy_scores[current] = entropy_score
            
            # Stop if entropy exceeds threshold or max depth reached
            if entropy_score > threshold or len(path) > max_depth:
                continue
                
            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return entropy_scores

def train_entropy_model(graph, embedding_dim=768, hidden_dim=256):
    """Train the entropy prediction model"""
    # Prepare training data
    X_train = []
    y_train = []
    
    for node in graph.nodes:
        # Get all paths from this node within same sentence
        sent_idx = graph.nodes[node]['sentence']
        same_sentence_nodes = [n for n in graph.nodes 
                              if graph.nodes[n]['sentence'] == sent_idx]
        
        # Generate positive examples (same sentence)
        for other in same_sentence_nodes:
            if node != other:
                path = nx.shortest_path(graph, node, other)
                path_embs = [graph.nodes[n]['embedding'] for n in path]
                X_train.append(path_embs)
                y_train.append(0.0)  # Low entropy for same sentence
        
        # Generate negative examples (different sentence)
        diff_sentence_nodes = [n for n in graph.nodes 
                              if graph.nodes[n]['sentence'] != sent_idx]
        if diff_sentence_nodes:
            other = np.random.choice(diff_sentence_nodes)
            try:
                path = nx.shortest_path(graph, node, other)
                path_embs = [graph.nodes[n]['embedding'] for n in path]
                X_train.append(path_embs)
                y_train.append(1.0)  # High entropy for different sentence
            except:
                pass
    
    # Pad sequences to same length
    max_len = max(len(x) for x in X_train)
    for i in range(len(X_train)):
        pad_len = max_len - len(X_train[i])
        X_train[i] += [np.zeros(embedding_dim)] * pad_len
    
    # Convert to tensors
    X_tensor = torch.tensor(np.array(X_train)).float()
    y_tensor = torch.tensor(np.array(y_train)).float()
    
    # Initialize model
    model = GraphEntropyModel(embedding_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model

def evaluate_model(model, graph):
    """Evaluate model performance"""
    true_labels = []
    pred_labels = []
    
    for node in graph.nodes:
        sent_idx = graph.nodes[node]['sentence']
        same_sentence = set(n for n in graph.nodes 
                           if graph.nodes[n]['sentence'] == sent_idx)
        
        # Get model predictions
        entropy_scores = processor.entropy_guided_traversal(node, model)
        predicted_set = set(n for n, score in entropy_scores.items() 
                           if score < 0.5)
        
        # Calculate metrics
        true_labels.append(list(same_sentence))
        pred_labels.append(list(predicted_set))
    
    # Calculate F1 score
    f1 = f1_score(
        [1 if n in true_set else 0 for true_set in true_labels for n in graph.nodes],
        [1 if n in pred_set else 0 for pred_set in pred_labels for n in graph.nodes],
        average='macro'
    )
    print(f"Model F1 Score: {f1:.4f}")
    return f1

# Example usage
if __name__ == "__main__":
    # Sample text from War and Peace
    text = """
    "Well, Prince, so Genoa and Lucca are now just family estates of the 
    Buonapartes. But I warn you, if you don't tell me that this means war, 
    if you still try to defend the infamies and horrors perpetrated by that 
    Antichrist- I really believe he is Antichrist- I will have nothing more 
    to do with you and you are no longer my friend, no longer my 'faithful 
    slave,' as you call yourself!"
    """
    
    # Initialize processor
    processor = KnowledgeGraphProcessor()
    kg = processor.build_graph(text)
    
    # Train entropy model
    entropy_model = train_entropy_model(kg)
    
    # Evaluate model
    evaluate_model(entropy_model, kg)
    
    # Perform entropy-guided traversal
    start_node = "prince"
    if start_node in kg.nodes:
        entropy_scores = processor.entropy_guided_traversal(start_node, entropy_model)
        print(f"\nEntropy scores for start node '{start_node}':")
        for node, entropy in entropy_scores.items():
            print(f"{node}: {entropy:.4f}")