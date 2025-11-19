from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager

from database import initialize_firebase, get_chroma_collection, fetch_job_complete_data, extract_estimate_data, extract_schedule_data
from embedding_service import EmbeddingService
from rag_service import RAGService
from models import DocumentSource, QueryRequest, QueryResponse

# Enhanced models for new functionality
class EnhancedQueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 10
    data_types: Optional[List[str]] = None  # Filter by 'consumed', 'estimate', 'flooring_estimate', 'schedule'
    
class RowSpecificQueryRequest(BaseModel):
    question: str
    row_number: int
    job_id: Optional[str] = None  # Optional: limit to specific job
    n_results: Optional[int] = 5

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
    
    print("🚀 Starting Enhanced Construction RAG System with Row-Level Capabilities...")
    
    # Initialize Firebase
    initialize_firebase()
    
    # Initialize services
    embedding_service = EmbeddingService()
    rag_service = RAGService(embedding_service)
    
    print("✅ System ready with multi-granularity support!")
    print("   - Job-level summaries")
    print("   - Row-level estimate details")
    print("   - Flooring estimate rows")
    print("   - Schedule data")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Construction RAG API",
    description="RAG system with row-level granularity for construction estimates, schedules, and cost data",
    version="3.0.0",
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
        "message": "Enhanced Construction RAG API with Row-Level Granularity is running!",
        "version": "3.0.0",
        "features": [
            "Row-specific estimate queries",
            "Multi-granularity embeddings (job + row level)",
            "Allowance filtering",
            "Materials queries",
            "Enhanced metadata for filtering"
        ],
        "supported_data_types": ["consumed", "estimate", "flooring_estimate", "schedule"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "services": ["firebase", "chromadb", "openai"],
        "data_types_supported": ["consumed", "estimate", "flooring_estimate", "schedule"],
        "granularities": ["job", "row"]
    }

@app.post("/documents", response_model=Dict[str, Any])
async def add_document(document: DocumentSource):
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
    """Query the RAG system with optional data type filtering and row-level support"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 10,
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
    """Enhanced search with detailed response including row-level data"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 10,
            data_types=query.data_types
        )
        return {
            "question": query.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "relevant_chunks": result["chunks"],
            "data_types_found": result.get("data_types_found", []),
            "granularities_found": result.get("granularities_found", []),
            "row_numbers_found": result.get("row_numbers_found", []),
            "document_types_found": result.get("document_types_found", []),
            "total_results": len(result["chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-row-specific")
async def search_row_specific(query: RowSpecificQueryRequest):
    """Search specifically for a row number across estimates"""
    try:
        # Build question that includes row number context
        enhanced_question = f"For estimate row number {query.row_number}: {query.question}"
        
        # Query with row filtering
        result = await rag_service.query(
            question=enhanced_question,
            n_results=query.n_results or 5,
            data_types=["estimate", "flooring_estimate"]  # Only search estimate data
        )
        
        return {
            "question": query.question,
            "row_number": query.row_number,
            "answer": result["answer"],
            "sources": result["sources"],
            "row_numbers_found": result.get("row_numbers_found", []),
            "total_results": len(result["chunks"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    company_id: Optional[str] = None
):
    """Manually trigger comprehensive embedding generation with row-level support"""
    try:
        async def process_data():
            return await rag_service.process_firebase_data(company_id)
        
        background_tasks.add_task(process_data)
        return {
            "message": "Comprehensive embedding generation started in background",
            "company_id": company_id or "all_companies",
            "data_types": ["consumed", "estimate", "flooring_estimate", "schedule"],
            "granularities": ["job-level summaries", "row-level details"]
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
    """Get enhanced system statistics with row-level breakdown"""
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_available_jobs():
    """Get list of jobs available in the system with data type and granularity information"""
    try:
        jobs = await rag_service.get_available_jobs()
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-types")
async def get_data_types_summary():
    """Get summary of available data types including row-level counts"""
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
        row_counts = {}
        
        # Check for consumed data
        if 'consumed_data' in job_data:
            available_data_types.append('consumed')
        
        # Check for estimate data
        estimate_data = extract_estimate_data(job_data)
        if estimate_data:
            available_data_types.append('estimate')
            row_counts['estimate'] = estimate_data.get('total_rows', 0)
        
        # Check for schedule data
        schedule_data = extract_schedule_data(job_data)
        if schedule_data:
            available_data_types.append('schedule')
            row_counts['schedule'] = schedule_data.get('total_rows', 0)
        
        return {
            "job_id": job_id,
            "company_id": company_id,
            "job_name": job_data.get('projectTitle', 'Unknown'),
            "available_data_types": available_data_types,
            "row_counts": row_counts,
            "client_name": job_data.get('clientName', ''),
            "site_location": f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-job-data/{company_id}/{job_id}")
async def debug_comprehensive_job_processing(company_id: str, job_id: str):
    """Debug how a specific job is being processed across all data types with row-level details"""
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
        
        # Process estimate data with row-level details
        estimate_data = extract_estimate_data(job_data)
        if estimate_data:
            entries = estimate_data.get('entries', [])
            sample_rows = []
            
            # Get first 3 rows as samples
            for entry in entries[:3]:
                sample_rows.append({
                    "row_number": entry.get('row_number'),
                    "area": entry.get('area'),
                    "task_scope": entry.get('taskScope'),
                    "description": entry.get('description', '')[:100],
                    "total": entry.get('total'),
                    "row_type": entry.get('rowType')
                })
            
            debug_info["data_type_processing"]["estimate"] = {
                "total_rows": estimate_data.get('total_rows', 0),
                "sample_rows": sample_rows,
                "job_level_text_preview": embedding_service.create_job_text_representation(estimate_data)[:500] + "...",
                "job_level_metadata": embedding_service.create_metadata(estimate_data),
                "row_level_example": {
                    "text_preview": embedding_service.create_estimate_row_text(
                        entries[0], 
                        {
                            'job_name': estimate_data.get('job_name'),
                            'site_location': estimate_data.get('site_location'),
                            'client_name': estimate_data.get('client_name')
                        }
                    )[:500] + "..." if entries else "No rows",
                    "metadata": embedding_service.create_estimate_row_metadata(
                        entries[0],
                        {
                            'job_name': estimate_data.get('job_name'),
                            'company_id': estimate_data.get('company_id'),
                            'job_id': estimate_data.get('job_id'),
                            'client_name': estimate_data.get('client_name'),
                            'site_location': estimate_data.get('site_location')
                        }
                    ) if entries else None
                }
            }
        
        # Process schedule data
        schedule_data = extract_schedule_data(job_data)
        if schedule_data:
            debug_info["data_type_processing"]["schedule"] = {
                "total_rows": schedule_data.get('total_rows', 0),
                "text_preview": embedding_service.create_job_text_representation(schedule_data)[:500] + "...",
                "metadata_sample": embedding_service.create_metadata(schedule_data)
            }
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-suggestions")
async def get_search_suggestions():
    """Get suggested search queries based on available data with row-level examples"""
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
        "estimate_row_queries": [
            "What's in estimate row 5?",
            "Show me details for row 10 of the estimate",
            "What are the materials in row 3?",
            "Compare rows 7 and 8",
            "Show all allowance rows",
            "List rows with materials"
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
        ],
        "tips": [
            "For row-specific queries, mention the row number explicitly (e.g., 'row 5', 'line 10')",
            "To see allowances, use keywords like 'allowance' or 'contingency'",
            "For materials, ask about 'materials' or 'items needed'",
            "You can combine filters: 'Show me allowance rows with materials'"
        ]
    }

@app.get("/query-examples")
async def get_query_examples():
    """Get comprehensive examples of different query types supported"""
    return {
        "row_specific_examples": [
            {
                "query": "What's in estimate row 5?",
                "description": "Get complete details about a specific estimate row including description, quantities, rates, and materials"
            },
            {
                "query": "Show me the materials for row 10",
                "description": "List all materials associated with a specific estimate row"
            },
            {
                "query": "What's the cost breakdown for row 3?",
                "description": "See estimated vs budgeted costs for a specific row"
            }
        ],
        "allowance_examples": [
            {
                "query": "Show me all allowance rows",
                "description": "List all items marked as allowances across estimates"
            },
            {
                "query": "What's the total allowance amount?",
                "description": "Sum up all allowance costs"
            },
            {
                "query": "Which areas have allowances?",
                "description": "See which project areas include allowance items"
            }
        ],
        "materials_examples": [
            {
                "query": "What materials are needed for electrical work?",
                "description": "List all materials for electrical scope"
            },
            {
                "query": "Show estimate rows that have materials",
                "description": "Find rows with material breakdowns"
            }
        ],
        "comparison_examples": [
            {
                "query": "Compare row 5 and row 8 costs",
                "description": "Side-by-side comparison of two estimate rows"
            },
            {
                "query": "Which rows are over budget?",
                "description": "Find rows where estimated cost exceeds budgeted amount"
            }
        ],
        "area_based_examples": [
            {
                "query": "Show me all rows in the Kitchen area",
                "description": "Filter rows by specific area"
            },
            {
                "query": "What's the total cost for Bathroom work?",
                "description": "Sum costs for a specific area"
            }
        ]
    }

@app.post("/generate-embeddings-job")
async def trigger_embedding_generation_single_job(
    company_id: str = Query(..., description="Company ID"),
    job_id: str = Query(..., description="Job ID"),
    background_tasks: BackgroundTasks = None
):
    """Generate embeddings for a single specific job"""
    try:
        async def process_single_job():
            from database import fetch_job_complete_data, extract_estimate_data, extract_flooring_estimate_data, extract_schedule_data
            from datetime import datetime
            
            print(f"🔄 Processing single job: {company_id}/{job_id}")
            
            # Fetch the specific job
            job_data = await fetch_job_complete_data(company_id, job_id)
            
            if not job_data:
                return {"error": f"Job {job_id} not found"}
            
            stats = {
                'job_id': job_id,
                'company_id': company_id,
                'datasets_embedded': 0,
                'rows_embedded': 0,
                'data_types_processed': []
            }
            
            documents_to_add = []
            embeddings_to_add = []
            ids_to_add = []
            metadatas_to_add = []
            
            # Process consumed data if exists
            if 'consumed_data' in job_data:
                consumed_data = job_data['consumed_data']
                text = embedding_service.create_job_text_representation(consumed_data)
                embedding = embedding_service.generate_embedding(text)
                metadata = embedding_service.create_metadata(consumed_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                doc_id = f"{company_id}_{job_id}_consumed_{timestamp}"
                
                documents_to_add.append(text)
                embeddings_to_add.append(embedding)
                ids_to_add.append(doc_id)
                metadatas_to_add.append(metadata)
                
                stats['datasets_embedded'] += 1
                stats['data_types_processed'].append('consumed')
                print(f"✅ Processed consumed data")
            
            # Process estimate data if exists
            estimate_data = extract_estimate_data(job_data)
            if estimate_data:
                # Job-level summary
                summary_text = embedding_service.create_job_text_representation(estimate_data)
                summary_embedding = embedding_service.generate_embedding(summary_text)
                summary_metadata = embedding_service.create_metadata(estimate_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_id = f"{company_id}_{job_id}_estimate_summary_{timestamp}"
                
                documents_to_add.append(summary_text)
                embeddings_to_add.append(summary_embedding)
                ids_to_add.append(summary_id)
                metadatas_to_add.append(summary_metadata)
                
                stats['datasets_embedded'] += 1
                
                # Row-level details
                entries = estimate_data.get('entries', [])
                job_context = {
                    'job_name': estimate_data.get('job_name'),
                    'company_id': company_id,
                    'job_id': job_id,
                    'client_name': estimate_data.get('client_name', ''),
                    'site_location': estimate_data.get('site_location', ''),
                    'last_updated': estimate_data.get('last_updated', ''),
                    'estimate_type': estimate_data.get('estimate_type', 'general')
                }
                
                for entry in entries:
                    row_num = entry.get('row_number', 0)
                    row_text = embedding_service.create_estimate_row_text(entry, job_context)
                    row_embedding = embedding_service.generate_embedding(row_text)
                    row_metadata = embedding_service.create_estimate_row_metadata(entry, job_context)
                    
                    row_id = f"{company_id}_{job_id}_estimate_row_{row_num}_{timestamp}"
                    
                    documents_to_add.append(row_text)
                    embeddings_to_add.append(row_embedding)
                    ids_to_add.append(row_id)
                    metadatas_to_add.append(row_metadata)
                    
                    stats['rows_embedded'] += 1
                
                stats['data_types_processed'].append('estimate')
                print(f"✅ Processed estimate: 1 summary + {len(entries)} rows")
            
            # Process flooring estimate if exists
            flooring_estimate_data = extract_flooring_estimate_data(job_data)
            if flooring_estimate_data:
                # Job-level summary
                summary_text = embedding_service.create_job_text_representation(flooring_estimate_data)
                summary_embedding = embedding_service.generate_embedding(summary_text)
                summary_metadata = embedding_service.create_metadata(flooring_estimate_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_id = f"{company_id}_{job_id}_flooring_summary_{timestamp}"
                
                documents_to_add.append(summary_text)
                embeddings_to_add.append(summary_embedding)
                ids_to_add.append(summary_id)
                metadatas_to_add.append(summary_metadata)
                
                stats['datasets_embedded'] += 1
                
                # Row-level details
                entries = flooring_estimate_data.get('entries', [])
                job_context = {
                    'job_name': flooring_estimate_data.get('job_name'),
                    'company_id': company_id,
                    'job_id': job_id,
                    'client_name': flooring_estimate_data.get('client_name', ''),
                    'site_location': flooring_estimate_data.get('site_location', '')
                }
                
                for entry in entries:
                    row_num = entry.get('row_number', 0)
                    row_text = embedding_service.create_flooring_estimate_row_text(entry, job_context)
                    row_embedding = embedding_service.generate_embedding(row_text)
                    row_metadata = embedding_service.create_flooring_estimate_row_metadata(entry, job_context)
                    
                    row_id = f"{company_id}_{job_id}_flooring_row_{row_num}_{timestamp}"
                    
                    documents_to_add.append(row_text)
                    embeddings_to_add.append(row_embedding)
                    ids_to_add.append(row_id)
                    metadatas_to_add.append(row_metadata)
                    
                    stats['rows_embedded'] += 1
                
                stats['data_types_processed'].append('flooring_estimate')
                print(f"✅ Processed flooring estimate: 1 summary + {len(entries)} rows")
            
            # Process schedule if exists
            schedule_data = extract_schedule_data(job_data)
            if schedule_data:
                text = embedding_service.create_job_text_representation(schedule_data)
                embedding = embedding_service.generate_embedding(text)
                metadata = embedding_service.create_metadata(schedule_data)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                doc_id = f"{company_id}_{job_id}_schedule_{timestamp}"
                
                documents_to_add.append(text)
                embeddings_to_add.append(embedding)
                ids_to_add.append(doc_id)
                metadatas_to_add.append(metadata)
                
                stats['datasets_embedded'] += 1
                stats['data_types_processed'].append('schedule')
                print(f"✅ Processed schedule data")
            
            # Add all documents to ChromaDB
            if documents_to_add:
                print(f"📝 Adding {len(documents_to_add)} documents to ChromaDB...")
                rag_service.collection.add(
                    documents=documents_to_add,
                    embeddings=embeddings_to_add,
                    ids=ids_to_add,
                    metadatas=metadatas_to_add
                )
                print(f"✅ Successfully added all documents")
            
            return stats
        
        if background_tasks:
            background_tasks.add_task(process_single_job)
            return {
                "message": "Single job embedding generation started in background",
                "company_id": company_id,
                "job_id": job_id
            }
        else:
            stats = await process_single_job()
            return {
                "message": "Single job embedding generation completed",
                "stats": stats
            }
            
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