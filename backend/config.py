import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Folder settings
    UPLOAD_FOLDER = "uploads"          # Regular temporary uploads
    KB_FOLDER = "knowledge_base"       # NEW: Permanent Knowledge Base folder
    CHROMA_DB_PATH = "./chroma_db"
    
    # Model settings
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama3.2"
    YOLO_MODEL = "yolov8n.pt"
    
    # Server settings
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    
    # OCR settings
    OCR_DPI = int(os.getenv("OCR_DPI", 300))
    OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", 50))
    
    # Chunking settings optimized for NATO doctrine
    # Doctrine chapters are 10-15 pages with 5-10 line paragraphs
    # Paragraphs ~100-200 chars, so we need overlapping chunks to preserve context
    CHUNK_SMALL = 300      # ~2 paragraphs for precise retrieval
    CHUNK_MEDIUM = 800     # ~4-5 paragraphs for balanced context
    CHUNK_LARGE = 1500     # ~8-10 paragraphs for full contextual understanding

# Ensure folders exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.KB_FOLDER, exist_ok=True)
os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)