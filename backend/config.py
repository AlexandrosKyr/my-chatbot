import os
from pathlib import Path
from dotenv import load_dotenv

# Explicit path so .env loads correctly regardless of working directory.
load_dotenv(Path(__file__).parent / ".env")

class Config:
    """Application configuration"""
    
    # Folder settings
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # backend/
    _DATA_DIR = os.path.join(_BASE_DIR, "..", "data")               # data/

    UPLOAD_FOLDER = os.path.join(_BASE_DIR, "uploads")
    KB_FOLDER = os.path.join(_DATA_DIR, "knowledge_base")
    CHROMA_DB_PATH = os.path.join(_DATA_DIR, "chroma_db")
    PARENT_CHUNKS_DB_PATH = os.path.join(_DATA_DIR, "chroma_db", "parent_chunks.db")
    
    # Model settings
    EMBEDDINGS_MODEL = "BAAI/bge-large-en-v1.5"  # Better retrieval for technical documents
    LLM_MODEL = "qwen3:8b-q4_K_M"
    
    # Server settings
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5001))
    
    # OCR settings
    OCR_DPI = int(os.getenv("OCR_DPI", 300))
    OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", 50))

    MIN_CHUNK_CHARS = 75   # Minimum characters for a valid chunk (filters garbage)

    # Hierarchical chunking settings
    PARENT_CHUNK_SIZE = 1200    # Large chunks sent to LLM as context
    PARENT_CHUNK_OVERLAP = 100
    CHILD_CHUNK_SIZE = 300      # Small chunks embedded for precise retrieval
    CHILD_CHUNK_OVERLAP = 50

    # Retrieval settings
    MIN_RELEVANCE_SCORE = 0.5  # Minimum cosine similarity for chunk retrieval

# Ensure runtime folders exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.KB_FOLDER, exist_ok=True)
os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)