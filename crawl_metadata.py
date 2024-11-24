import os
import json
import arxiv
from multiprocessing import Pool, cpu_count


def fetch_metadata_for_arxiv_id(args):
    """
    Fetch metadata for a single arXiv ID and save it as a JSON file.
    """
    arxiv_id, folder_path = args
    if arxiv_id == 'auto' or arxiv_id == 'images':
        return
    metadata_file_path = os.path.join(folder_path, f"{arxiv_id}_metadata.json")

    # Skip if the metadata file already exists
    if os.path.exists(metadata_file_path):
        print(f"Metadata already exists for {arxiv_id}, skipping...")
        return

    try:
        print(f"Fetching metadata for {arxiv_id}...")
        search = arxiv.Search(id_list=[arxiv_id])
        result = next(search.results())  # Fetch the first (and only) result

        # Extract relevant metadata
        metadata = {
            "id": result.entry_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
            "doi": result.doi,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "journal_reference": result.journal_ref,
        }

        # Save metadata as a JSON file
        with open(metadata_file_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        print(f"Metadata saved to {metadata_file_path}")

    except StopIteration:
        print(f"No metadata found for {arxiv_id}, skipping...")
    except Exception as e:
        print(f"Error while fetching metadata for {arxiv_id}: {e}")


def parallel_process_arxiv_metadata(base_dirs, num_workers):
    """
    Process arXiv metadata extraction using multiple processes.
    """
    # Collect all arXiv IDs and their folder paths
    arxiv_tasks = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for folder in dirs:
                arxiv_id = folder
                folder_path = os.path.join(root, folder)
                arxiv_tasks.append((arxiv_id, folder_path))

    print(f"Collected {len(arxiv_tasks)} arXiv IDs for processing.")

    # Use a multiprocessing Pool to fetch metadata in parallel
    with Pool(processes=num_workers) as pool:
        pool.map(fetch_metadata_for_arxiv_id, arxiv_tasks)


if __name__ == "__main__":
    # Base directories for your processed folders
    base_dirs = [
        "../neurips2021_processed",
        "../neurips2022_processed",
        "../neurips2023_processed"
    ]
    
    # Number of worker processes (use all available CPUs)
    num_workers = 6

    # Start parallel processing
    print(f"Starting metadata fetch with {num_workers} workers...")
    parallel_process_arxiv_metadata(base_dirs, num_workers)