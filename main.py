from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager

from app.database import initialize_firebase, get_chroma_collection
from app.embedding_service import EmbeddingService
from app.rag_service import RAGService
from app.scheduler import start_scheduler
from app.models import DocumentCreate, QueryRequest, QueryResponse

# Global services
embedding_service = None
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_service, rag_service
    
    print("🚀 Starting Construction RAG System...")
    
    # Initialize Firebase
    initialize_firebase()
    
    # Initialize services
    embedding_service = EmbeddingService()
    rag_service = RAGService(embedding_service)
    
    # Start nightly scheduler
    start_scheduler()
    
    print("✅ System ready!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Construction RAG API",
    description="RAG system for construction job data queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Construction RAG API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": ["firebase", "chromadb", "openai"]}

@app.post("/documents", response_model=Dict[str, Any])
async def add_document(document: DocumentCreate):
    """Manually add a document (for testing or manual data entry)"""
    try:
        doc_id = await rag_service.add_document(
            text=document.text,
            metadata=document.metadata
        )
        return {"success": True, "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=QueryResponse)
async def search_documents(query: QueryRequest):
    """Query the RAG system"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 5
        )
        return QueryResponse(
            question=query.question,
            answer=result["answer"],
            sources=result["sources"],
            relevant_chunks=result["chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def trigger_embedding_generation(background_tasks: BackgroundTasks):
    """Manually trigger embedding generation (for testing)"""
    try:
        background_tasks.add_task(rag_service.process_firebase_data)
        return {"message": "Embedding generation started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        collection = get_chroma_collection()
        count = collection.count()
        
        # Get some sample data structure info
        sample_data = collection.peek(limit=3)
        
        return {
            "total_documents": count,
            "sample_metadata": sample_data.get("metadatas", []) if sample_data else [],
            "embedding_model": "text-embedding-3-small"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_available_jobs():
    """Get list of jobs available in the system"""
    try:
        jobs = await rag_service.get_available_jobs()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost-codes")
async def get_available_cost_codes():
    """Get list of cost codes found in the data"""
    try:
        cost_codes = await rag_service.get_available_cost_codes()
        return {"cost_codes": cost_codes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True
    )