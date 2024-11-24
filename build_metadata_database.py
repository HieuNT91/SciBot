import os
import json
import sqlite3

def create_database(db_path="database/full_neurips_metadata.db"):
    """
    Create an SQLite database with a schema to store NeurIPS metadata.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table for the metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        abstract TEXT,
        published DATE,
        updated DATE,
        doi TEXT,
        primary_category TEXT,
        categories TEXT,
        journal_reference TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def populate_database(folder_path, db_path="database/full_neurips_metadata.db"):
    """
    Populate the SQLite database with JSON files from the specified folder.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Iterate over each JSON file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                    
                    # Extract relevant fields
                    paper_id = data.get("id")
                    title = data.get("title")
                    authors = ", ".join(data.get("authors", []))  # Join authors into a single string
                    abstract = data.get("abstract")
                    published = data.get("published")
                    updated = data.get("updated")
                    doi = data.get("doi")
                    primary_category = data.get("primary_category")
                    categories = ", ".join(data.get("categories", []))  # Join categories into a single string
                    journal_reference = data.get("journal_reference")
                    
                    # Insert the data into the database
                    cursor.execute("""
                    INSERT OR IGNORE INTO papers (
                        id, title, authors, abstract, published, updated, doi, 
                        primary_category, categories, journal_reference
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (paper_id, title, authors, abstract, published, updated, doi, primary_category, categories, journal_reference))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")

    conn.commit()
    conn.close()

# Create and populate the database
create_database()
populate_database("../full_neurips_metadata")  # Replace with the actual path to your folder