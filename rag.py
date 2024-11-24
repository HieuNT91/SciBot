import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer

# Path to your processed data
output_folder = "../processed_full_neurips"
index_path = f"{output_folder}/faiss_index_flatip"
metadata_path = f"{output_folder}/metadata.json"

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# Load metadata
print("Loading metadata...")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load Sentence Transformer model for query embedding
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


def query_faiss_index(query, model, index, metadata, top_k=5):
    """
    Retrieve the most relevant chunks from the FAISS index.

    Args:
        query (str): User query.
        model (SentenceTransformer): Model for embedding queries.
        index (faiss.Index): FAISS index.
        metadata (list): Metadata corresponding to the FAISS index.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: Top-k relevant chunks and their metadata.
    """
    # Generate query embedding
    query_embedding = model.encode(query, convert_to_tensor=False)
    faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize for cosine similarity

    # Search in FAISS
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    # Retrieve metadata for the top-k results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            result = metadata[idx]
            result["score"] = dist
            results.append(result)

    return results

def generate_response_with_ollama(query, retrieved_chunks, model_name="llama3.2"):

    context = "\n\n".join([chunk["chunk"] for chunk in retrieved_chunks])
    # Construct the input prompt
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    # API request payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Send POST request to Ollama (streaming response)
    try:
        with requests.post(
            "http://localhost:11434/api/chat",  # Ollama's chat API endpoint
            json=payload,
            stream=True  # Enable streaming
        ) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)

            # Stream and accumulate the response
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Ignore keep-alive newlines
                    try:
                        # Parse the JSON chunk
                        chunk_data = json.loads(line)

                        # Extract the "content" from "message"
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            content = chunk_data["message"]["content"]
                            full_response += content  # Accumulate the response
                            print(content, end="", flush=True)  # Stream in real-time
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON line: {line}")

            return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return None


def rag_pipeline(query, model, index, metadata, top_k=5, ollama_model="llama3.2"):
    """
    Run the RAG pipeline: retrieve relevant chunks and generate a response.

    Args:
        query (str): User query.
        model (SentenceTransformer): Model for embedding queries.
        index (faiss.Index): FAISS index.
        metadata (list): Metadata corresponding to the FAISS index.
        top_k (int): Number of top results to retrieve.
        ollama_model (str): Name of the Ollama model to use.

    Returns:
        str: Generated response from the model.
    """
    # Step 1: Retrieve relevant chunks
    print("Retrieving relevant chunks from FAISS...")
    retrieved_chunks = query_faiss_index(query, model, index, metadata, top_k=top_k)

    # Step 2: Generate response with Ollama
    print("Generating response with Ollama...")
    response = generate_response_with_ollama(query, retrieved_chunks, model_name=ollama_model)

    return response


# Example usage
if __name__ == "__main__":
    query = "Can you explain Reinforcement Learning to me ?"
    print(f"User Query: {query}")

    response = rag_pipeline(query, model, index, metadata, top_k=5, ollama_model="llama3.1:70b")
