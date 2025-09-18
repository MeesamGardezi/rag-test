from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager

from database import initialize_firebase, get_chroma_collection
from embedding_service import EmbeddingService
from rag_service import RAGService
from models import DocumentCreate, QueryRequest, QueryResponse

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
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    company_id: Optional[str] = None
):
    """Manually trigger embedding generation"""
    try:
        async def process_data():
            return await rag_service.process_firebase_data(company_id)
        
        background_tasks.add_task(process_data)
        return {
            "message": "Embedding generation started in background",
            "company_id": company_id or "all_companies"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings-sync")
async def trigger_embedding_generation_sync(company_id: Optional[str] = None):
    """Manually trigger embedding generation (synchronous)"""
    try:
        stats = await rag_service.process_firebase_data(company_id)
        return {
            "message": "Embedding generation completed",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embeddings/clear")
async def clear_all_embeddings():
    """Clear all embeddings from ChromaDB"""
    try:
        collection = get_chroma_collection()
        all_data = collection.get()
        
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
            return {
                "message": f"Cleared {len(all_data['ids'])} documents",
                "cleared_count": len(all_data['ids'])
            }
        else:
            return {"message": "No documents to clear", "cleared_count": 0}
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

@app.get("/collection-stats")
async def get_collection_stats():
    """Get detailed collection statistics"""
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-firebase")
async def test_firebase_connection():
    """Test Firebase connection and credentials"""
    import os
    try:
        # Check if environment variables are loaded
        firebase_config = {
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "has_private_key": bool(os.getenv("FIREBASE_PRIVATE_KEY"))
        }
        
        # Try to get Firebase database
        db = get_firebase_db()
        
        # Try to list companies (just to test permissions)
        companies_ref = db.collection('companies').limit(1)
        companies = list(companies_ref.stream())
        
        return {
            "credentials_loaded": firebase_config,
            "firebase_connected": True,
            "companies_found": len(companies),
            "test_successful": True
        }
    except Exception as e:
        return {
            "credentials_loaded": firebase_config,
            "firebase_connected": False,
            "error": str(e),
            "test_successful": False
        }

@app.get("/debug-job/{company_id}/{job_id}")
async def debug_job_processing(company_id: str, job_id: str):
    """Debug how a specific job is being processed"""
    from database import fetch_job_data
    
    try:
        # Get raw job data from Firebase
        raw_job_data = await fetch_job_data(company_id, job_id)
        
        if not raw_job_data:
            return {"error": "Job not found"}
        
        # Process it through embedding service
        text_representation = embedding_service.create_job_text_representation(raw_job_data)
        metadata = embedding_service.create_metadata(raw_job_data)
        
        # Categorize each entry to see what's happening
        entry_analysis = []
        total_by_category = {}
        
        for entry in raw_job_data.get('entries', []):
            cost_code = entry.get('costCode', '')
            amount_str = entry.get('amount', '0')
            
            try:
                amount = float(amount_str) if amount_str else 0.0
            except (ValueError, TypeError):
                amount = 0.0
            
            category = embedding_service.categorize_cost_code(cost_code)
            
            if category not in total_by_category:
                total_by_category[category] = 0.0
            total_by_category[category] += amount
            
            entry_analysis.append({
                'cost_code': cost_code,
                'amount_str': amount_str,
                'amount_float': amount,
                'category': category,
                'job_name': entry.get('job', '')
            })
        
        return {
            "raw_data": {
                "total_entries": len(raw_job_data.get('entries', [])),
                "last_updated": str(raw_job_data.get('lastUpdated', '')),
                "sample_entries": raw_job_data.get('entries', [])[:5]  # First 5 entries
            },
            "processing_analysis": {
                "category_totals": total_by_category,
                "entry_breakdown": entry_analysis
            },
            "generated_metadata": metadata,
            "text_representation_preview": text_representation[:500] + "..." if len(text_representation) > 500 else text_representation
        }
        
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