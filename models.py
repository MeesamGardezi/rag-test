from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
import json

# Enums for better type safety
class DataType(str, Enum):
    CONSUMED = "consumed"
    ESTIMATE = "estimate" 
    SCHEDULE = "schedule"
    DOCUMENT = "document"
    SPECIFICATION = "specification"

class IntentType(str, Enum):
    COST_ANALYSIS = "cost_analysis"
    SCHEDULE_QUERY = "schedule_query"
    TECHNICAL_SPECIFICATION = "technical_specification"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    RESPONSIBILITY_ASSIGNMENT = "responsibility_assignment"
    VARIANCE_ANALYSIS = "variance_analysis"
    BUDGET_COMPARISON = "budget_comparison"
    PROJECT_TIMELINE = "project_timeline"
    RESOURCE_ALLOCATION = "resource_allocation"
    CROSS_DATA_ANALYSIS = "cross_data_analysis"

class SearchStrategy(str, Enum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    SEMANTIC = "semantic"
    FULL_TEXT = "full_text"
    COLBERT = "colbert"

class ModelProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

# Core data models
class ConstructionEntity(BaseModel):
    """Extracted construction domain entity"""
    entity_type: str = Field(..., description="Type of entity (project, cost_code, material, etc.)")
    value: str = Field(..., description="Entity value")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    context: Optional[str] = Field(None, description="Surrounding context")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryIntent(BaseModel):
    """Classified query intent with routing information"""
    intent_type: IntentType = Field(..., description="Primary intent classification")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    secondary_intents: List[IntentType] = Field(default_factory=list, description="Secondary intents detected")
    suggested_data_types: List[DataType] = Field(default_factory=list, description="Recommended data types to query")
    requires_calculation: bool = Field(False, description="Whether automatic calculation is needed")
    requires_visualization: bool = Field(False, description="Whether data visualization is needed")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Query complexity for resource allocation")

class DocumentMetadata(BaseModel):
    """Enhanced metadata for construction documents"""
    # Core identification
    document_id: str = Field(..., description="Unique document identifier")
    data_type: DataType = Field(..., description="Type of construction data")
    
    # Project context
    company_id: str = Field(..., description="Company identifier")
    job_id: str = Field(..., description="Job/Project identifier") 
    job_name: str = Field(..., description="Human readable job name")
    
    # Construction-specific metadata
    cost_categories: List[str] = Field(default_factory=list, description="Applicable cost categories")
    project_phases: List[str] = Field(default_factory=list, description="Related project phases")
    csi_codes: List[str] = Field(default_factory=list, description="CSI MasterFormat codes")
    uniformat_codes: List[str] = Field(default_factory=list, description="UniFormat codes")
    
    # Financial information
    total_amount: Optional[float] = Field(None, ge=0, description="Total monetary amount")
    currency: str = Field("USD", description="Currency code")
    
    # Temporal information  
    document_date: Optional[datetime] = Field(None, description="Document creation date")
    last_updated: Optional[datetime] = Field(None, description="Last modification date")
    project_start_date: Optional[datetime] = Field(None, description="Project start date")
    project_end_date: Optional[datetime] = Field(None, description="Project end date")
    
    # Quality and status
    data_quality_score: float = Field(1.0, ge=0.0, le=1.0, description="Data quality assessment")
    completion_status: str = Field("active", description="Project completion status")
    approval_status: str = Field("approved", description="Document approval status")
    
    # Location and client information
    project_location: Optional[str] = Field(None, description="Project location")
    client_name: Optional[str] = Field(None, description="Client name")
    contractor_name: Optional[str] = Field(None, description="Primary contractor")
    
    # Additional context
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

class EmbeddingResult(BaseModel):
    """Embedding generation result with metadata"""
    vector: List[float] = Field(..., description="Dense vector embedding")
    model_used: str = Field(..., description="Model used for embedding")
    embedding_dimension: int = Field(..., description="Vector dimension")
    
    # ColBERT support
    token_embeddings: Optional[List[List[float]]] = Field(None, description="Token-level embeddings for ColBERT")
    
    # Matryoshka support
    reduced_vectors: Optional[Dict[int, List[float]]] = Field(None, description="Reduced dimension vectors")
    
    # Generation metadata
    generation_time_ms: float = Field(..., description="Embedding generation time in milliseconds")
    chunk_size: int = Field(..., description="Size of text chunk embedded")

class SearchResult(BaseModel):
    """Individual search result with rich metadata"""
    document_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Retrieved text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Scoring information
    vector_score: float = Field(..., ge=0.0, le=1.0, description="Vector similarity score")
    sparse_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sparse retrieval score")
    full_text_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Full-text search score") 
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final combined score")
    
    # Ranking information
    rank: int = Field(..., ge=1, description="Result ranking position")
    reranked: bool = Field(False, description="Whether result was reranked")
    
    # Retrieval metadata
    retrieval_strategy: SearchStrategy = Field(..., description="Strategy used for retrieval")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")

class AnalysisInsight(BaseModel):
    """Generated insight from cross-data analysis"""
    insight_type: str = Field(..., description="Type of insight (variance, trend, anomaly, etc.)")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    confidence: float = Field(ge=0.0, le=1.0, description="Insight confidence level")
    
    # Supporting data
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Data supporting the insight")
    visualizations: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested visualizations")
    
    # Business impact
    impact_level: Literal["low", "medium", "high", "critical"] = Field("medium", description="Business impact level")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    
    # Context
    affected_projects: List[str] = Field(default_factory=list, description="Affected project IDs")
    timeframe: Optional[str] = Field(None, description="Relevant timeframe")

class CalculationResult(BaseModel):
    """Automatic calculation result"""
    calculation_type: str = Field(..., description="Type of calculation performed")
    result: Dict[str, Any] = Field(..., description="Calculation results")
    units: Optional[str] = Field(None, description="Units of measurement")
    
    # Methodology
    formula_used: Optional[str] = Field(None, description="Formula or method used")
    assumptions: List[str] = Field(default_factory=list, description="Calculation assumptions")
    
    # Validation
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Statistical confidence interval")
    data_quality_factors: List[str] = Field(default_factory=list, description="Factors affecting quality")
    
    # Context
    input_data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    calculation_time: datetime = Field(default_factory=datetime.utcnow, description="When calculation was performed")

# Request/Response models
class EnhancedQueryRequest(BaseModel):
    """Advanced query request with full configuration"""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    
    # Search configuration
    data_types: Optional[List[DataType]] = Field(None, description="Data types to search")
    n_results: int = Field(10, ge=1, le=50, description="Number of results to return")
    search_strategy: SearchStrategy = Field(SearchStrategy.HYBRID, description="Search strategy to use")
    
    # Context and filtering
    project_filters: Optional[Dict[str, Any]] = Field(None, description="Project-specific filters")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filters")
    amount_range: Optional[Dict[str, float]] = Field(None, description="Amount range filters")
    
    # Advanced features
    enable_calculation: bool = Field(True, description="Enable automatic calculations")
    enable_insights: bool = Field(True, description="Enable insight generation")
    enable_visualization: bool = Field(False, description="Enable visualization suggestions")
    
    # Session context
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation turns")
    
    # User context
    user_role: Optional[str] = Field(None, description="User role (project_manager, estimator, etc.)")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

    @validator('conversation_history')
    def validate_conversation_history(cls, v):
        for turn in v:
            if not isinstance(turn, dict) or 'question' not in turn or 'answer' not in turn:
                raise ValueError("Conversation history must contain dicts with 'question' and 'answer' keys")
        return v

class EnhancedQueryResponse(BaseModel):
    """Comprehensive query response with analysis results"""
    # Basic response
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    
    # Query processing metadata
    intent: QueryIntent = Field(..., description="Detected query intent")
    entities: List[ConstructionEntity] = Field(default_factory=list, description="Extracted entities")
    
    # Search results
    search_results: List[SearchResult] = Field(default_factory=list, description="Retrieved search results")
    total_results_found: int = Field(..., description="Total results found")
    
    # Analysis results
    calculations: List[CalculationResult] = Field(default_factory=list, description="Automatic calculations")
    insights: List[AnalysisInsight] = Field(default_factory=list, description="Generated insights")
    
    # Recommendations
    related_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    
    # Data sources and confidence
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall response confidence")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")
    retrieval_time_ms: float = Field(..., description="Retrieval time")
    generation_time_ms: float = Field(..., description="Answer generation time")
    
    # Conversation context
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_turn: int = Field(1, description="Turn number in conversation")

class DocumentCreateRequest(BaseModel):
    """Request to create/add new document"""
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    
    # Processing options
    auto_chunk: bool = Field(True, description="Automatically chunk document")
    chunk_strategy: str = Field("semantic", description="Chunking strategy")
    generate_embeddings: bool = Field(True, description="Generate embeddings immediately")
    
    # Quality control
    skip_validation: bool = Field(False, description="Skip content validation")
    merge_with_existing: bool = Field(False, description="Merge with existing document if duplicate")

class SystemStats(BaseModel):
    """Comprehensive system statistics"""
    # Data statistics
    total_documents: int = Field(..., description="Total documents in system")
    documents_by_type: Dict[DataType, int] = Field(default_factory=dict, description="Document counts by type")
    total_projects: int = Field(..., description="Total projects tracked")
    total_companies: int = Field(..., description="Total companies")
    
    # Embedding statistics
    total_embeddings: int = Field(..., description="Total embeddings generated")
    embedding_model: str = Field(..., description="Current embedding model")
    average_embedding_time_ms: float = Field(..., description="Average embedding generation time")
    
    # Query statistics
    total_queries: int = Field(..., description="Total queries processed")
    average_query_time_ms: float = Field(..., description="Average query processing time")
    most_common_intents: List[Dict[str, Any]] = Field(default_factory=list, description="Most common query intents")
    
    # System health
    system_health: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system health")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Stats last updated time")
    
    # Performance metrics
    cache_hit_rate: float = Field(ge=0.0, le=1.0, description="Cache hit rate")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate over last hour")
    
    # Resource utilization
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    storage_usage_mb: float = Field(..., description="Storage usage in MB")

class BatchProcessingJob(BaseModel):
    """Batch processing job for large-scale operations"""
    job_id: str = Field(..., description="Unique job identifier")
    job_type: str = Field(..., description="Type of batch job")
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = Field(..., description="Job status")
    
    # Progress tracking
    total_items: int = Field(..., ge=0, description="Total items to process")
    completed_items: int = Field(0, ge=0, description="Items completed")
    failed_items: int = Field(0, ge=0, description="Items that failed")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Configuration and results
    job_config: Dict[str, Any] = Field(default_factory=dict, description="Job configuration parameters")
    results: Dict[str, Any] = Field(default_factory=dict, description="Job results")
    error_messages: List[str] = Field(default_factory=list, description="Error messages")
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100.0

class ConfigurationUpdate(BaseModel):
    """Configuration update request"""
    config_section: str = Field(..., description="Configuration section to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    validate_before_apply: bool = Field(True, description="Validate configuration before applying")
    restart_required: bool = Field(False, description="Whether restart is required")
    
    # Change tracking
    updated_by: Optional[str] = Field(None, description="User making the update")
    update_reason: Optional[str] = Field(None, description="Reason for update")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")

# Legacy compatibility models (maintaining backward compatibility)
class DocumentSource(BaseModel):
    """Legacy document source model for backward compatibility"""
    job_name: str
    company_id: str
    job_id: str
    cost_code: Optional[str] = None
    amount: Optional[str] = None
    last_updated: Optional[str] = None

class QueryRequest(BaseModel):
    """Legacy query request model for backward compatibility"""
    question: str = Field(..., description="The question to ask")
    n_results: Optional[int] = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    """Legacy query response model for backward compatibility"""
    question: str
    answer: str
    sources: List[DocumentSource]
    relevant_chunks: List[str]

# Utility functions for model conversion
def convert_to_legacy_source(metadata: DocumentMetadata, content: str = "") -> DocumentSource:
    """Convert new metadata model to legacy DocumentSource"""
    return DocumentSource(
        job_name=metadata.job_name,
        company_id=metadata.company_id,
        job_id=metadata.job_id,
        cost_code=", ".join(metadata.cost_categories) if metadata.cost_categories else None,
        amount=f"${metadata.total_amount:,.2f}" if metadata.total_amount else None,
        last_updated=metadata.last_updated.isoformat() if metadata.last_updated else None
    )

def convert_to_legacy_response(enhanced_response: EnhancedQueryResponse) -> QueryResponse:
    """Convert enhanced response to legacy format"""
    sources = []
    relevant_chunks = []
    
    for result in enhanced_response.search_results:
        sources.append(convert_to_legacy_source(result.metadata, result.content))
        relevant_chunks.append(result.content)
    
    return QueryResponse(
        question=enhanced_response.question,
        answer=enhanced_response.answer,
        sources=sources,
        relevant_chunks=relevant_chunks
    )