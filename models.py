from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentCreate(BaseModel):
    """Model for creating a new document"""
    text: str = Field(..., description="The text content to embed")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for the document")

class QueryRequest(BaseModel):
    """Model for RAG query requests"""
    question: str = Field(..., description="The question to ask about the construction data")
    n_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to return")

class DocumentSource(BaseModel):
    """Model for document sources in query responses"""
    job_name: str
    company_id: str
    job_id: str
    cost_code: Optional[str] = None
    amount: Optional[str] = None
    last_updated: Optional[str] = None

class QueryResponse(BaseModel):
    """Model for RAG query responses"""
    question: str
    answer: str
    sources: List[DocumentSource]
    relevant_chunks: List[str]

class JobCostEntry(BaseModel):
    """Model for individual cost entries"""
    job: str
    cost_code: str  
    amount: str
    category: str  # Material, Labor, Subcontractor, Other
    
class ProcessedJobData(BaseModel):
    """Model for processed job data from Firebase"""
    company_id: str
    job_id: str
    job_name: str
    entries: List[JobCostEntry]
    last_updated: str
    total_entries: int

class EmbeddingStats(BaseModel):
    """Model for embedding generation statistics"""
    total_jobs_processed: int
    total_entries_embedded: int
    companies_processed: List[str]
    processing_time_seconds: float
    errors: List[str]
    
class SystemHealth(BaseModel):
    """Model for system health checks"""
    firebase_connected: bool
    chromadb_connected: bool
    openai_connected: bool
    last_embedding_run: Optional[datetime] = None
    total_documents: int
    
class JobSummary(BaseModel):
    """Model for job summary information"""
    job_name: str
    job_id: str
    total_cost: float
    entry_count: int
    major_categories: List[str]
    last_updated: str

class CostCodeSummary(BaseModel):
    """Model for cost code summary"""
    cost_code: str
    description: str
    category: str
    total_jobs_used: int
    average_amount: float