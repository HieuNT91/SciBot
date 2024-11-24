import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, index_path, metadata_path, model_name="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = None

    def load_resources(self):
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(self.model_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        print("Loading metadata...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("Resources loaded successfully.")

    def unload_resources(self):
        print("Unloading resources...")
        self.model = None
        self.index = None
        self.metadata = None
        print("Resources unloaded.")

    def query(self, query_text, top_k=5):
        if not self.model or not self.index or not self.metadata:
            raise RuntimeError("Resources are not loaded. Please load them first.")

        query_embedding = self.model.encode(query_text, convert_to_tensor=False)
        faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize for cosine similarity
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx]
                result["score"] = dist
                results.append(result)

        return results

    # Context manager methods
    def __enter__(self):
        self.load_resources()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload_resources()