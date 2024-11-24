# File counts per folder:
# neurips2021_processed: 1530 .md files
# neurips2022_processed: 1893 .md files
# neurips2023_processed: 2374 .md files
# Total .md files copied: 5797

import os
import shutil

# Define the source folders
source_folders = ["../neurips2021_processed", "../neurips2022_processed", "../neurips2023_processed"]

# Define the destination folder
# HIEU: UNCOMMENT THIS LINE
destination_folder = "../full_neurips_md"
# destination_folder = "../full_neurips_metadata"
limit = -1
# Create the destination folder if it doesn't already exist
os.makedirs(destination_folder, exist_ok=True)

# Initialize counters for each folder and the total
file_counts = {folder: 0 for folder in source_folders}
total_files = 0

# Loop through each source folder
for source_folder in source_folders:
    # Walk through the folder structure
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is a .md file
            # HIEU: UNCOMMENT THIS LINE
            # if file.endswith("_metadata.json"):
            if file.endswith(".md"):
                # Construct the full file path
                source_file_path = os.path.join(root, file)
                # Construct the destination file path
                destination_file_path = os.path.join(destination_folder, file)

                # Copy the .md file to the destination folder
                shutil.copy2(source_file_path, destination_file_path)

                # Increment the count for the current source folder
                file_counts[source_folder] += 1
                # Increment the total file count
                total_files += 1
            if total_files >= limit and limit != -1:
                break
        if total_files >= limit and limit != -1:
            break

# Report the results
print("File counts per folder:")
for folder, count in file_counts.items():
    print(f"{folder}: {count} .md files")

print(f"Total .md files copied: {total_files}")
