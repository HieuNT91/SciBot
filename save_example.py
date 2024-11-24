from pypdf import PdfReader

# Load the PDF file
reader = PdfReader("../neurips2023_1/2305.16380v4.pdf")

# Open a Markdown (.md) file to save the extracted content
with open("example_pypdf.md", "w", encoding="utf-8") as md_file:
    for i, page in enumerate(reader.pages):
        # Extract text from the page
        text = page.extract_text()
        
        # Write to the Markdown file with a page header
        md_file.write(f"# Page {i + 1}\n\n")
        md_file.write(text.strip() + "\n\n")
        md_file.write("---\n\n")  # Add a horizontal rule between pages

print("Text extraction complete. Check 'example_pypdf'.")