import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    UPLOAD_FOLDER = "uploads"
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
    
    # Chunking settings
    CHUNK_SMALL = 200
    CHUNK_MEDIUM = 500
    CHUNK_LARGE = 1000

# Ensure folders exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)