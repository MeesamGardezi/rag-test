from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager

from database import initialize_firebase, get_chroma_collection, fetch_job_complete_data, extract_estimate_data, extract_schedule_data
from embedding_service import EmbeddingService
from rag_service import RAGService
from models import DocumentCreate, QueryRequest, QueryResponse

# Enhanced models for new functionality
class EnhancedQueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5
    data_types: Optional[List[str]] = None  # Filter by 'consumed', 'estimate', 'schedule'
    
class DataTypeFilterRequest(BaseModel):
    company_id: Optional[str] = None
    data_types: Optional[List[str]] = None

# Global services
embedding_service = None
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_service, rag_service
    
    print("🚀 Starting Enhanced Construction RAG System...")
    
    # Initialize Firebase
    initialize_firebase()
    
    # Initialize services
    embedding_service = EmbeddingService()
    rag_service = RAGService(embedding_service)
    
    print("✅ System ready with multi-data type support!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Construction RAG API",
    description="RAG system for construction job data queries with consumed, estimate, and schedule data",
    version="2.0.0",
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
    return {
        "message": "Enhanced Construction RAG API is running!",
        "version": "2.0.0",
        "supported_data_types": ["consumed", "estimate", "schedule"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "services": ["firebase", "chromadb", "openai"],
        "data_types_supported": ["consumed", "estimate", "schedule"]
    }

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
async def search_documents(query: EnhancedQueryRequest):
    """Query the RAG system with optional data type filtering"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 5,
            data_types=query.data_types
        )
        return QueryResponse(
            question=query.question,
            answer=result["answer"],
            sources=result["sources"],
            relevant_chunks=result["chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-enhanced")
async def search_documents_enhanced(query: EnhancedQueryRequest):
    """Enhanced search with detailed response including data types found"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 5,
            data_types=query.data_types
        )
        return {
            "question": query.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "relevant_chunks": result["chunks"],
            "data_types_found": result.get("data_types_found", []),
            "document_types_found": result.get("document_types_found", []),
            "total_results": len(result["chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    company_id: Optional[str] = None
):
    """Manually trigger comprehensive embedding generation (consumed, estimate, schedule)"""
    try:
        async def process_data():
            return await rag_service.process_firebase_data(company_id)
        
        background_tasks.add_task(process_data)
        return {
            "message": "Comprehensive embedding generation started in background",
            "company_id": company_id or "all_companies",
            "data_types": ["consumed", "estimate", "schedule"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings-sync")
async def trigger_embedding_generation_sync(company_id: Optional[str] = None):
    """Manually trigger comprehensive embedding generation (synchronous)"""
    try:
        stats = await rag_service.process_firebase_data(company_id)
        return {
            "message": "Comprehensive embedding generation completed",
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
    """Get enhanced system statistics"""
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_available_jobs():
    """Get list of jobs available in the system with data type information"""
    try:
        jobs = await rag_service.get_available_jobs()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-types")
async def get_data_types_summary():
    """Get summary of available data types in the system"""
    try:
        summary = await rag_service.get_data_types_summary()
        return {"data_types_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost-codes")
async def get_available_cost_codes():
    """Get list of cost codes and areas found in the data"""
    try:
        cost_codes = await rag_service.get_available_cost_codes()
        return {"cost_codes": cost_codes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{company_id}/{job_id}")
async def get_job_data_types(company_id: str, job_id: str):
    """Get available data types for a specific job"""
    try:
        job_data = await fetch_job_complete_data(company_id, job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        available_data_types = []
        
        # Check for consumed data
        if 'consumed_data' in job_data:
            available_data_types.append('consumed')
        
        # Check for estimate data
        estimate_data = extract_estimate_data(job_data)
        if estimate_data:
            available_data_types.append('estimate')
        
        # Check for schedule data
        schedule_data = extract_schedule_data(job_data)
        if schedule_data:
            available_data_types.append('schedule')
        
        return {
            "job_id": job_id,
            "company_id": company_id,
            "job_name": job_data.get('projectTitle', 'Unknown'),
            "available_data_types": available_data_types,
            "client_name": job_data.get('clientName', ''),
            "site_location": f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-job-data/{company_id}/{job_id}")
async def debug_comprehensive_job_processing(company_id: str, job_id: str):
    """Debug how a specific job is being processed across all data types"""
    try:
        # Get complete job data from Firebase
        job_data = await fetch_job_complete_data(company_id, job_id)
        
        if not job_data:
            return {"error": "Job not found"}
        
        debug_info = {
            "raw_job_metadata": {
                "project_title": job_data.get('projectTitle', ''),
                "client_name": job_data.get('clientName', ''),
                "created_date": str(job_data.get('createdDate', '')),
                "has_consumed_data": 'consumed_data' in job_data,
                "has_estimate_array": 'estimate' in job_data and isinstance(job_data['estimate'], list),
                "has_schedule_array": 'schedule' in job_data and isinstance(job_data['schedule'], list)
            },
            "data_type_processing": {}
        }
        
        # Process consumed data if exists
        if 'consumed_data' in job_data:
            consumed_data = job_data['consumed_data']
            debug_info["data_type_processing"]["consumed"] = {
                "entries_count": len(consumed_data.get('entries', [])),
                "text_preview": embedding_service.create_job_text_representation(consumed_data)[:500] + "...",
                "metadata_sample": embedding_service.create_metadata(consumed_data)
            }
        
        # Process estimate data
        estimate_data = extract_estimate_data(job_data)
        if estimate_data:
            debug_info["data_type_processing"]["estimate"] = {
                "entries_count": len(estimate_data.get('entries', [])),
                "text_preview": embedding_service.create_job_text_representation(estimate_data)[:500] + "...",
                "metadata_sample": embedding_service.create_metadata(estimate_data)
            }
        
        # Process schedule data
        schedule_data = extract_schedule_data(job_data)
        if schedule_data:
            debug_info["data_type_processing"]["schedule"] = {
                "entries_count": len(schedule_data.get('entries', [])),
                "text_preview": embedding_service.create_job_text_representation(schedule_data)[:500] + "...",
                "metadata_sample": embedding_service.create_metadata(schedule_data)
            }
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-suggestions")
async def get_search_suggestions():
    """Get suggested search queries based on available data"""
    return {
        "consumed_data_queries": [
            "What was spent on electrical work?",
            "Show me all subcontractor costs",
            "Which job had the highest material costs?",
            "Total consumed costs across all projects"
        ],
        "estimate_data_queries": [
            "What's the estimated cost for plumbing work?",
            "Compare estimated vs budgeted costs",
            "Show me estimates by area",
            "Which estimates have the highest variance?"
        ],
        "schedule_data_queries": [
            "What tasks are scheduled for next month?",
            "Show me project timelines",
            "Which tasks are behind schedule?",
            "Resource allocation across projects"
        ],
        "cross_data_queries": [
            "Compare estimated vs consumed costs for Project X",
            "Show me schedule and cost data for electrical work",
            "Which projects are over budget and behind schedule?"
        ]
    }

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
        from database import get_firebase_db
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True
    )