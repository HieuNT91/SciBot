from setuptools import setup, find_packages

setup(
    name="scibot",
    version="0.1.0",
    description="A semantic search tool using FAISS and Sentence Transformers",
    author="Hieu Trung Nguyen",
    author_email="hilljun.2000@gmail.com",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu",
        "numpy",
        "sentence-transformers",
    ],
    entry_points={
        "console_scripts": [
            "scibot=scibot.cli:main",  # Point to 'scibot.cli:main'
        ],
    },
    python_requires=">=3.10",
)