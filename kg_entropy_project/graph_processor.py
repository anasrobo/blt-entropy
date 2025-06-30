import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, BertModel
import torch

class KnowledgeGraphProcessor:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = BertModel.from_pretrained('bert-base-uncased')
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.embedder(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def build_graph(self, text):
        sentences = sent_tokenize(text)
        for sent_idx, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            
            # Simple triplet extraction (customize as needed)
            for i, (word, pos) in enumerate(tagged):
                if pos.startswith('VB'):  # Verb
                    subj = " ".join([w for w, p in tagged[:i] if p in ['NN', 'NNS']])
                    obj = " ".join([w for w, p in tagged[i+1:] if p in ['NN', 'NNS']])
                    
                    if subj and obj:
                        subj_emb = self.get_embedding(subj)
                        obj_emb = self.get_embedding(obj)
                        
                        self.graph.add_node(subj, embedding=subj_emb, sentence=sent_idx)
                        self.graph.add_node(obj, embedding=obj_emb, sentence=sent_idx)
                        self.graph.add_edge(subj, obj, predicate=word)
        
        return self.graph