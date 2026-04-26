import logging

logger = logging.getLogger(__name__)

_retriever = None


def initialize(retriever) -> None:
    """Set the DoctrineRetriever instance. Call once at startup."""
    global _retriever
    _retriever = retriever


def retrieve_doctrine(query: str, terrain_data: dict = None) -> str:
    """Search the doctrine knowledge base. Returns cited passages or empty string."""
    if _retriever is None:
        return ""
    return _retriever.retrieve(query, terrain_data)
