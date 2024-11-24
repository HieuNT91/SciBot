import faiss
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import os 

# Load embeddings
output_folder = "../processed_full_neurips"
embeddings_path = os.path.join(output_folder, "embeddings.npy")

embeddings = np.load(embeddings_path)  # Load embeddings
faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity

# Parameters
num_queries = 100  # Number of synthetic queries
top_k = 5  # Number of top results to retrieve
dimension = embeddings.shape[1]  # Dimensionality of embeddings
num_embeddings = len(embeddings)  # Number of embeddings
nlist = 100  # Number of clusters for IVF

# Step 1: Randomly select synthetic queries from the dataset
print("\n=== Selecting Synthetic Queries ===")
np.random.seed(42)
query_indices = np.random.choice(len(embeddings), num_queries, replace=False)  # Randomly sample indices
synthetic_queries = embeddings[query_indices]
print(f"Selected {num_queries} synthetic queries from the dataset.")

# Function to benchmark query time
def benchmark_query_time(index, queries, top_k):
    """
    Benchmark the average query time for a FAISS index.
    """
    query_times = []
    for query in queries:
        start_time = time.time()
        distances, indices = index.search(query.reshape(1, -1), top_k)
        query_time = time.time() - start_time
        query_times.append(query_time)
    avg_query_time = np.mean(query_times)
    return avg_query_time

# Function to compute recall
def compute_recall(index, embeddings, queries, query_indices, top_k):
    """
    Compute recall by comparing FAISS results with brute-force cosine similarity.
    """
    total_recall = 0
    for i, query in enumerate(queries):
        # Get FAISS results
        distances_faiss, indices_faiss = index.search(query.reshape(1, -1), top_k)

        # Compute brute-force cosine similarity
        cosine_sim = cosine_similarity(query.reshape(1, -1), embeddings)[0]
        top_k_brute_force = np.argsort(-cosine_sim)[:top_k]

        # Compare FAISS results with brute-force results
        recall = len(set(indices_faiss[0]) & set(top_k_brute_force)) / top_k
        total_recall += recall

    average_recall = total_recall / len(queries)
    return average_recall

# Benchmark 1: IndexFlat (Exact Search)
print("\n=== Benchmarking IndexFlat (Exact Search) ===")
index_flat = faiss.IndexFlatIP(dimension)  # Exact brute-force index

start_time = time.time()
index_flat.add(embeddings)  # Add embeddings to the index
index_flat_creation_time = time.time() - start_time
print(f"IndexFlat created with {index_flat.ntotal} embeddings in {index_flat_creation_time:.4f} seconds.")

flat_avg_query_time = benchmark_query_time(index_flat, synthetic_queries, top_k)
flat_recall = compute_recall(index_flat, embeddings, synthetic_queries, query_indices, top_k)
print(f"IndexFlat Average Query Time: {flat_avg_query_time:.6f} seconds")
print(f"IndexFlat Recall: {flat_recall:.4f}")

# Benchmark 2: IndexHNSWFlat (HNSW)
print("\n=== Benchmarking IndexHNSWFlat (HNSW) ===")
M = 32  # Number of bi-directional links per node
ef_construction = 200  # Controls the construction time/accuracy tradeoff
index_hnsw = faiss.IndexHNSWFlat(dimension, M)  # HNSW index

index_hnsw.hnsw.efConstruction = ef_construction  # Set efConstruction
start_time = time.time()
index_hnsw.add(embeddings)  # Add embeddings to the HNSW index
index_hnsw_creation_time = time.time() - start_time
print(f"IndexHNSWFlat created with {index_hnsw.ntotal} embeddings in {index_hnsw_creation_time:.4f} seconds.")

# Set search parameter for HNSW
ef_search = 50  # Controls the tradeoff between speed and recall during search
index_hnsw.hnsw.efSearch = ef_search

hnsw_avg_query_time = benchmark_query_time(index_hnsw, synthetic_queries, top_k)
hnsw_recall = compute_recall(index_hnsw, embeddings, synthetic_queries, query_indices, top_k)
print(f"IndexHNSWFlat Average Query Time: {hnsw_avg_query_time:.6f} seconds")
print(f"IndexHNSWFlat Recall: {hnsw_recall:.4f}")

# Benchmark 3: IndexOPQ (Optimized Product Quantization)
print("\n=== Benchmarking IndexOPQ ===")
m = 16  # Number of subspaces for PQ
opq = faiss.OPQMatrix(dimension, m)  # Optimized PQ transformation
index_opq = faiss.IndexPreTransform(opq, faiss.IndexFlatL2(dimension))  # Combine OPQ with Flat index

start_time = time.time()
index_opq.train(embeddings)  # Train OPQ with embeddings
index_opq.add(embeddings)  # Add embeddings to the index
index_opq_creation_time = time.time() - start_time
print(f"IndexOPQ created with {index_opq.ntotal} embeddings in {index_opq_creation_time:.4f} seconds.")

opq_avg_query_time = benchmark_query_time(index_opq, synthetic_queries, top_k)
opq_recall = compute_recall(index_opq, embeddings, synthetic_queries, query_indices, top_k)
print(f"IndexOPQ Average Query Time: {opq_avg_query_time:.6f} seconds")
print(f"IndexOPQ Recall: {opq_recall:.4f}")

# Benchmark 4: IVF + IndexOPQ (Clustering + OPQ)
print("\n=== Benchmarking IVF + IndexOPQ ===")
m = 16  # Number of subspaces for PQ
opq = faiss.OPQMatrix(dimension, m)  # Optimized PQ transformation
quantizer = faiss.IndexFlatL2(dimension)  # Coarse quantizer
index_ivf_opq = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)  # IVF Index

index_ivf_opq = faiss.IndexPreTransform(opq, index_ivf_opq)  # Add OPQ to IVF
start_time = time.time()
index_ivf_opq.train(embeddings)  # Train OPQ and IVF
index_ivf_opq.add(embeddings)  # Add embeddings to the index
index_ivf_opq_creation_time = time.time() - start_time
print(f"IVF + IndexOPQ created with {index_ivf_opq.ntotal} embeddings in {index_ivf_opq_creation_time:.4f} seconds.")

# Set search parameters for IVF
index_ivf_opq.nprobe = 10  # Number of clusters to search

ivf_opq_avg_query_time = benchmark_query_time(index_ivf_opq, synthetic_queries, top_k)
ivf_opq_recall = compute_recall(index_ivf_opq, embeddings, synthetic_queries, query_indices, top_k)
print(f"IVF + IndexOPQ Average Query Time: {ivf_opq_avg_query_time:.6f} seconds")
print(f"IVF + IndexOPQ Recall: {ivf_opq_recall:.4f}")

# Final Results
print("\n=== Final Benchmark Results ===")
print(f"IndexFlat Creation Time: {index_flat_creation_time:.4f} seconds")
print(f"IndexFlat Average Query Time: {flat_avg_query_time:.6f} seconds")
print(f"IndexFlat Recall: {flat_recall:.4f}")
print(f"IndexHNSWFlat Creation Time: {index_hnsw_creation_time:.4f} seconds")
print(f"IndexHNSWFlat Average Query Time: {hnsw_avg_query_time:.6f} seconds")
print(f"IndexHNSWFlat Recall: {hnsw_recall:.4f}")
print(f"IndexOPQ Creation Time: {index_opq_creation_time:.4f} seconds")
print(f"IndexOPQ Average Query Time: {opq_avg_query_time:.6f} seconds")
print(f"IndexOPQ Recall: {opq_recall:.4f}")
print(f"IVF + IndexOPQ Creation Time: {index_ivf_opq_creation_time:.4f} seconds")
print(f"IVF + IndexOPQ Average Query Time: {ivf_opq_avg_query_time:.6f} seconds")
print(f"IVF + IndexOPQ Recall: {ivf_opq_recall:.4f}")