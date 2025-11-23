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
    LLM_MODEL = "llama3.2:latest"  # Lightweight model (2GB) for better performance on laptops
    YOLO_MODEL = "yolov8n.pt"
    
    # Server settings
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5001))  # Changed from 5000 to avoid macOS AirPlay conflict
    
    # OCR settings
    OCR_DPI = int(os.getenv("OCR_DPI", 300))
    OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", 50))
    
    CHUNK_SMALL = 300      # ~2 paragraphs for precise retrieval
    CHUNK_MEDIUM = 800     # ~4-5 paragraphs for balanced context
    CHUNK_LARGE = 1500     # ~8-10 paragraphs for full contextual understanding

# Ensure folders exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.KB_FOLDER, exist_ok=True)
os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)