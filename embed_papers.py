import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Initialize Sentence Transformer Model
model_name = "all-mpnet-base-v2"  # Lightweight and efficient
model = SentenceTransformer(model_name)

# Define Input and Output Paths
input_folder_md = "../full_neurips_md"  # Path to Markdown files
input_folder_json = "../full_neurips_metadata"  # Path to JSON metadata files
output_folder = "../processed_full_neurips_mpnet"  # Path to save processed embeddings and metadata
os.makedirs(output_folder, exist_ok=True)

# Maximum token size for Sentence Transformer (e.g., 512 tokens)
MAX_TOKENS = 256

# Function to read and process Markdown files
def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Function to read JSON metadata
def read_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to split text into chunks (semantic chunking)
def chunk_text(text, model, max_tokens=MAX_TOKENS):
    sentences = text.split(". ")  # Split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_length = len(model.tokenizer.encode(sentence))  # Estimate token count
        if current_length + token_length > max_tokens:  # Start a new chunk if limit exceeded
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_length
        else:
            current_chunk.append(sentence)
            current_length += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Batch Processing
def process_files(md_folder, json_folder, model, output_dir, batch_size=100):
    md_files = sorted(Path(md_folder).glob("*.md"))
    json_files = sorted(Path(json_folder).glob("*.json"))

    embeddings = []
    metadata_list = []

    for i in tqdm(range(0, len(md_files), batch_size), desc="Processing Batches"):
        batch_md_files = md_files[i:i + batch_size]
        batch_json_files = json_files[i:i + batch_size]

        for md_file, json_file in zip(batch_md_files, batch_json_files):
            # Read Markdown and JSON data
            md_content = read_markdown(md_file)
            metadata = read_metadata(json_file)

            # Split Markdown content into chunks
            chunks = chunk_text(md_content, model)

            # Generate embeddings for each chunk
            for chunk in chunks:
                embedding = model.encode(chunk, convert_to_tensor=False)
                embeddings.append(embedding)

                # Save metadata for the chunk
                metadata_list.append({
                    "paper_id": metadata.get("id"),
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors"),
                    "primary_category": metadata.get("primary_category"),
                    "published": metadata.get("published"),
                    "section": None,  # Can add logic to extract section headers if needed
                    "chunk": chunk
                })

    # Save embeddings and metadata to output files
    np.save(os.path.join(output_dir, "embeddings.npy"), np.array(embeddings))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"Processed {len(metadata_list)} chunks. Saved embeddings and metadata.")

# Run Batch Processing
process_files(input_folder_md, input_folder_json, model, output_folder)