import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from textual.app import App, ComposeResult
from textual.widgets import Input, Button, Static, TextLog, Header, Footer
from textual.containers import Vertical

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


# Textual-based UI App
class RAGSearchApp(App):
    CSS = """
    Screen {
        align: center middle;
    }

    #title {
        margin-top: 1;
        margin-bottom: 1;
        text-align: center;
        background: $surface;
        padding: 1;
    }

    #results {
        margin-top: 2;
        padding: 1;
        background: $surface;
        width: 80%;
        height: 50%;
    }

    Input {
        width: 80%;
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 2;
    }
    """

    def compose(self) -> ComposeResult:
        # Header and input components
        yield Header()
        yield Static("RAG-based Semantic Search", id="title")
        yield Input(placeholder="Enter your query here...", id="query_input")
        yield Button("Search", id="search_button")
        yield TextLog(id="results", highlight=True, wrap=True)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search_button":
            query_input = self.query_one("#query_input", Input).value
            if query_input.strip():
                self.search_query(query_input)

    def search_query(self, query: str) -> None:
        # Perform the query using the provided `query_index` function
        results = query_index(query, model, index, metadata, top_k=5)

        # Display the results in the results TextLog
        results_view = self.query_one("#results", TextLog)
        results_view.clear()  # Clear the previous results

        if results:
            for i, result in enumerate(results):
                title = result.get("title", "No Title")
                chunk = result.get("chunk", "No Content")
                score = result.get("score", "N/A")
                results_view.write(
                    f"[b]Rank {i + 1}[/b]\n"
                    f"Title: {title}\n"
                    f"Chunk: {chunk}\n"
                    f"Score: {score:.4f}\n"
                )
        else:
            results_view.write("No results found for your query.")


if __name__ == "__main__":
    app = RAGSearchApp()
    app.run()