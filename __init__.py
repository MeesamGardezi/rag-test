# Construction RAG System
# A RAG (Retrieval Augmented Generation) system for construction job cost data

__version__ = "1.0.0"
__author__ = "Construction RAG Team"
__description__ = "RAG system for querying construction job cost data stored in Firebase"

# Import key components for easier access
from .database import initialize_firebase, get_chroma_collection, get_firebase_db
from .embedding_service import EmbeddingService  
from .rag_service import RAGService

__all__ = [
    "initialize_firebase",
    "get_chroma_collection", 
    "get_firebase_db",
    "EmbeddingService",
    "RAGService"
]