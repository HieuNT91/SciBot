import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Paths to the saved database files
output_folder = "../processed_full_neurips"    # Folder where the FAISS index and metadata are stored
index_path = os.path.join(output_folder, "faiss_index_flatip")  # Path to the FAISS index file
embeddings_path = os.path.join(output_folder, "embeddings.npy")  # Path to the embeddings file
metadata_path = os.path.join(output_folder, "metadata.json")  # Path to the metadata file

# Load the Sentence Transformer model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# Load metadata
print("Loading metadata...")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embeddings (optional, not strictly needed for queries but useful for debugging)
print("Loading embeddings...")
embeddings = np.load(embeddings_path)

# Function to query the FAISS index
def query_index(query, model, index, metadata, top_k=5):
    """
    Query the FAISS index and retrieve the top-k most similar results.

    Args:
        query (str): The search query in natural language.
        model (SentenceTransformer): The Sentence Transformer model for embedding generation.
        index (faiss.Index): The FAISS index containing document embeddings.
        metadata (list): The metadata corresponding to the embeddings.
        top_k (int): The number of top results to retrieve.

    Returns:
        list: A list of dictionaries containing metadata and similarity scores for the top-k results.
    """
    # Generate query embedding
    query_embedding = model.encode(query, convert_to_tensor=False)
    faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize query embedding for cosine similarity

    # Search the FAISS index
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    # Retrieve results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            result = metadata[idx]
            result["score"] = dist
            results.append(result)

    return results

# Example Query
query = "The full definition of Reinforcement Learning"
print(f"Query: {query}")

# Perform the query
results = query_index(query, model, index, metadata, top_k=5)

# Display Results
print("\nTop Results:")
for i, result in enumerate(results):
    print(f"Rank {i+1}:")
    print(f"Title: {result['title']}")
    print(f"Chunk: {result['chunk']}")  # Truncate chunk for display
    print(f"Score: {result['score']}")
    print()