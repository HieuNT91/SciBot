import os
import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    
class AskPipeline:
    def __init__(self, model=None, index=None, metadata=None, ollama_model="llama3.2"):
        """
        Initializes the AskPipeline with preloaded resources.

        Args:
            model (SentenceTransformer): Preloaded SentenceTransformer model.
            index (faiss.Index): Preloaded FAISS index.
            metadata (list): Preloaded metadata as a list.
            ollama_model (str): Name of the Ollama model to use.
        """
        self.model = model
        self.index = index
        self.metadata = metadata
        self.ollama_model = ollama_model

    def query_faiss_index(self, query_text, top_k=5):
        """
        Retrieve the most relevant chunks from the FAISS index.

        Args:
            query_text (str): User query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Top-k relevant chunks and their metadata.
        """
        if not self.model or not self.index or not self.metadata:
            raise RuntimeError("Resources are not loaded. Please provide preloaded resources.")

        # Generate query embedding
        query_embedding = self.model.encode(query_text, convert_to_tensor=False)
        faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normalize for cosine similarity

        # Search in FAISS
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        # Retrieve metadata for the top-k results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx]
                result["score"] = dist
                results.append(result)

        return results

    def generate_response_with_ollama(self, query_text, retrieved_chunks):
        """
        Generate a response using the Ollama API.

        Args:
            query_text (str): User query.
            retrieved_chunks (list): Retrieved chunks from the FAISS index.

        Returns:
            str: Generated response from the Ollama model.
        """
        if not retrieved_chunks:
            return "No relevant chunks found to generate a response."

        # Build context from retrieved chunks
        context = "\n\n".join([chunk["chunk"] for chunk in retrieved_chunks])

        # Construct the input prompt
        prompt = f"Context:\n{context}\n\nQuestion:\n{query_text}\n\nAnswer:"

        # API request payload
        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Send POST request to Ollama (streaming response)
        try:
            print("Connecting to Ollama API...")
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

    def generate_rag_response(self, query_text, top_k=5):
        """
        Run the RAG pipeline: retrieve relevant chunks and generate a response.

        Args:
            query_text (str): User query.
            top_k (int): Number of top results to retrieve.

        Returns:
            str: Generated response from the model.
        """
        print("Retrieving relevant chunks from FAISS...")
        retrieved_chunks = self.query_faiss_index(query_text, top_k=top_k)

        print("Generating response with Ollama...")
        response = self.generate_response_with_ollama(query_text, retrieved_chunks)

        return response