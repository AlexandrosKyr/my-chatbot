import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from config import Config
from utils import ParentChunkStore

logger = logging.getLogger(__name__)

class ModelLoader:
    """Initialize and manage all models"""

    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.parent_store = None
    
    def load_embeddings(self):
        """Load HuggingFace embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDINGS_MODEL)
            logger.info(f"Embeddings loaded: {Config.EMBEDDINGS_MODEL}")
            return self.embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None
    
    def load_llm(self):
        """Load Ollama LLM"""
        try:
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                num_ctx=16384,
            )
            logger.info(f"LLM loaded: {Config.LLM_MODEL}")
            return self.llm
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            return None

    def load_vectorstore(self, persist_directory=None):
        """Load or create vector store"""
        try:
            if persist_directory is None:
                persist_directory = Config.CHROMA_DB_PATH
            
            if self.embeddings is None:
                logger.warning("Embeddings not loaded, cannot initialize vector store")
                return None
            
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info(f"Vector store loaded from {persist_directory}")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
    
    def check_ollama_connection(self):
        """Test Ollama connection with a lightweight API ping (no LLM inference)"""
        if self.llm is None:
            return False, "LLM not initialized"
        try:
            import requests as req
            resp = req.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            return True, "Ollama is running"
        except Exception as e:
            return False, f"Ollama connection failed: {str(e)}"
    
    def check_embeddings(self):
        """Test embeddings"""
        if self.embeddings is None:
            return False, "Embeddings not initialized"
        try:
            self.embeddings.embed_query("test")
            return True, "Embeddings working"
        except Exception as e:
            return False, f"Embeddings failed: {str(e)}"
    
    def load_all(self):
        """Load all models"""
        logger.info("="*60)
        logger.info("INITIALIZING MODELS")
        logger.info("="*60)

        self.load_embeddings()
        self.load_llm()
        self.load_vectorstore()
        self.parent_store = ParentChunkStore(Config.PARENT_CHUNKS_DB_PATH)

        logger.info("="*60)
        logger.info("MODEL INITIALIZATION COMPLETE")
        logger.info("="*60)

        return self


# Global model loader instance
_model_loader = None

def get_models():
    """Get global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all()
    return _model_loader