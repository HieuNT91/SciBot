import faiss
import numpy as np
import json
import os 

output_folder = "../processed_full_neurips"  
# Load embeddings and metadata
embeddings_path = os.path.join(output_folder, "embeddings.npy")
metadata_path = os.path.join(output_folder, "metadata.json")

embeddings = np.load(embeddings_path)  # Load embeddings
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)  # Load metadata

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Create a FAISS index
dimension = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
index.add(embeddings)  # Add embeddings to the index

# Save FAISS index
faiss.write_index(index, os.path.join(output_folder, "faiss_index_flatip"))

print(f"FAISS index created with {index.ntotal} embeddings.")