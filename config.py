import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from watchfiles import awatch
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and strategies"""
    model_name: str = "text-embedding-3-small"
    model_provider: str = "openai"
    embedding_dimension: int = 1536
    batch_size: int = 100
    use_colbert: bool = False
    colbert_model: str = "colbert-ir/colbertv2.0"
    use_matryoshka: bool = True
    matryoshka_dimensions: List[int] = field(default_factory=lambda: [1536, 768, 512, 256, 128])
    chunk_size: int = 500
    chunk_overlap: int = 50
    domain_fine_tuned: bool = False
    fine_tuned_model_path: Optional[str] = None

@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    # ChromaDB settings
    chroma_persist_path: str = "./chroma_storage"
    chroma_collection_name: str = "construction_rag"
    chroma_memory_limit_gb: int = 16
    chroma_cache_policy: str = "LRU"
    
    # ChromaDB HNSW optimization
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 200
    hnsw_m: int = 32
    hnsw_search_ef: int = 100
    hnsw_batch_size: int = 500
    hnsw_sync_threshold: int = 2000
    
    # Firebase settings
    firebase_project_id: str = ""
    firebase_private_key: str = ""
    firebase_client_email: str = ""
    
    # Connection pooling
    db_pool_size: int = 20
    redis_pool_size: int = 10
    connection_timeout: int = 30

@dataclass
class LLMConfig:
    """Configuration for Language Model settings"""
    provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    context_window: int = 128000
    
    # Advanced generation settings
    use_structured_output: bool = True
    enable_function_calling: bool = True
    streaming: bool = False

@dataclass
class QueryRoutingConfig:
    """Configuration for intelligent query routing"""
    enable_intent_detection: bool = True
    intent_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    confidence_threshold: float = 0.7
    
    # Intent categories
    intent_categories: List[str] = field(default_factory=lambda: [
        "cost_analysis", "schedule_query", "technical_specification", 
        "regulatory_compliance", "responsibility_assignment", "variance_analysis",
        "budget_comparison", "project_timeline", "resource_allocation"
    ])
    
    # Data type routing
    auto_data_type_detection: bool = True
    default_data_types: List[str] = field(default_factory=lambda: ["consumed", "estimate", "schedule"])
    
    # Multi-hop reasoning
    enable_multi_hop: bool = True
    max_reasoning_steps: int = 3

@dataclass
class SearchConfig:
    """Configuration for hybrid search strategies"""
    # Search strategy weights
    vector_weight: float = 0.5
    sparse_weight: float = 0.3
    full_text_weight: float = 0.2
    
    # Search parameters
    n_results: int = 10
    diversity_threshold: float = 0.7
    rerank_top_k: int = 5
    
    # Advanced search features
    enable_hypothetical_questions: bool = True
    enable_query_expansion: bool = True
    enable_semantic_caching: bool = True
    
    # Construction-specific filters
    enable_metadata_filtering: bool = True
    default_filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConstructionRulesConfig:
    """Configuration for construction industry business rules"""
    # Cost categorization rules
    cost_categorization: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "material_keywords": ["material", "materials", "supply", "equipment"],
        "labor_keywords": ["labor", "labour", "worker", "crew", "manpower"],
        "subcontractor_keywords": ["subcontractor", "sub", "contractor"],
        "overhead_keywords": ["overhead", "management", "supervision", "permit", "fee"],
        
        "suffix_patterns": {
            "M": "Materials",
            "L": "Labor", 
            "S": "Subcontractors",
            "O": "Other/Overhead"
        },
        
        "priority_rules": [
            {"pattern": "subcontractor", "category": "Subcontractors"},
            {"pattern": r"\d+M\b", "category": "Materials"},
            {"pattern": r"\d+L\b", "category": "Labor"},
            {"pattern": r"\d+S\b", "category": "Subcontractors"},
            {"pattern": r"\d+O\b", "category": "Other/Overhead"}
        ]
    })
    
    # Schedule analysis rules
    schedule_rules: Dict[str, Any] = field(default_factory=lambda: {
        "critical_path_threshold": 0.1,
        "delay_warning_days": 3,
        "resource_conflict_threshold": 1.2,
        "progress_tracking_frequency": "weekly"
    })
    
    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=lambda: {
        "max_cost_variance": 0.15,  # 15%
        "min_schedule_buffer": 5,   # days
        "required_approvals": ["project_manager", "client"]
    })

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability"""
    # Logging settings
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    log_format: str = "json"
    
    # Metrics collection
    enable_prometheus: bool = True
    metrics_port: int = 8001
    
    # Tracing
    enable_tracing: bool = True
    jaeger_endpoint: Optional[str] = None
    
    # Performance monitoring
    track_query_performance: bool = True
    track_embedding_performance: bool = True
    track_retrieval_performance: bool = True

@dataclass
class AdvancedRAGConfig:
    """Main configuration class combining all settings"""
    # Core components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query_routing: QueryRoutingConfig = field(default_factory=QueryRoutingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    construction_rules: ConstructionRulesConfig = field(default_factory=ConstructionRulesConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    environment: str = "development"
    debug: bool = False
    version: str = "2.0.0"
    
    # API settings
    api_title: str = "Advanced Construction RAG API"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

class ConfigurationManager:
    """Advanced configuration manager with hot reloading and validation"""
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        self.config_paths = config_paths or [
            "config/base.yaml",
            "config/construction.yaml", 
            "config/environment.yaml"
        ]
        self.config: Optional[AdvancedRAGConfig] = None
        self.observers: List[callable] = []
        self.watching = False
    
    async def load_config(self) -> AdvancedRAGConfig:
        """Load configuration from multiple sources with precedence"""
        config_data = {}
        
        # Load from files
        for config_path in self.config_paths:
            if Path(config_path).exists():
                config_data.update(await self._load_config_file(config_path))
        
        # Override with environment variables
        config_data.update(self._load_env_overrides())
        
        # Create configuration object
        self.config = self._create_config_object(config_data)
        
        # Validate configuration
        await self._validate_config(self.config)
        
        logger.info(f"Configuration loaded successfully from {len(self.config_paths)} sources")
        return self.config
    
    async def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a single file"""
        try:
            path = Path(config_path)
            content = path.read_text()
            
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif path.suffix == '.json':
                return json.loads(content)
            else:
                logger.warning(f"Unsupported config file format: {config_path}")
                return {}
        except Exception as e:
            logger.warning(f"Could not load config file {config_path}: {e}")
            return {}
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        overrides = {}
        
        # Environment variable mapping
        env_mappings = {
            "OPENAI_API_KEY": ("llm.api_key", "embedding.api_key"),
            "EMBEDDING_MODEL": ("embedding.model_name",),
            "LLM_MODEL": ("llm.model_name",),
            "CHROMA_PERSIST_PATH": ("database.chroma_persist_path",),
            "FIREBASE_PROJECT_ID": ("database.firebase_project_id",),
            "ENVIRONMENT": ("environment",),
            "DEBUG": ("debug",),
            "API_PORT": ("api_port",)
        }
        
        for env_var, config_paths in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert boolean strings
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Convert numeric strings
                elif value.isdigit():
                    value = int(value)
                
                for config_path in config_paths:
                    self._set_nested_value(overrides, config_path, value)
        
        return overrides
    
    def _set_nested_value(self, dict_obj: Dict, path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = dict_obj
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> AdvancedRAGConfig:
        """Create configuration object from dictionary"""
        try:
            return AdvancedRAGConfig(**config_data)
        except Exception as e:
            logger.error(f"Error creating configuration object: {e}")
            # Return default configuration on error
            return AdvancedRAGConfig()
    
    async def _validate_config(self, config: AdvancedRAGConfig):
        """Validate configuration settings"""
        validations = []
        
        # Validate required API keys
        if not os.getenv("OPENAI_API_KEY"):
            validations.append("OPENAI_API_KEY environment variable is required")
        
        # Validate paths
        for path in [config.database.chroma_persist_path]:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validations.append(f"Cannot create directory {path}: {e}")
        
        # Validate model configurations
        if config.embedding.embedding_dimension <= 0:
            validations.append("Embedding dimension must be positive")
        
        # Validate search weights sum to 1.0
        total_weight = (config.search.vector_weight + 
                       config.search.sparse_weight + 
                       config.search.full_text_weight)
        if abs(total_weight - 1.0) > 0.001:
            validations.append(f"Search weights must sum to 1.0, got {total_weight}")
        
        if validations:
            raise ValueError(f"Configuration validation failed: {'; '.join(validations)}")
    
    async def start_watching(self):
        """Start watching configuration files for changes"""
        if self.watching:
            return
        
        self.watching = True
        
        async def watch_config_files():
            try:
                async for changes in awatch(*[Path(p).parent for p in self.config_paths]):
                    logger.info(f"Configuration files changed: {changes}")
                    await self.reload_config()
            except Exception as e:
                logger.error(f"Error watching configuration files: {e}")
        
        # Start watching in background
        asyncio.create_task(watch_config_files())
    
    async def reload_config(self):
        """Reload configuration and notify observers"""
        try:
            old_config = self.config
            await self.load_config()
            
            # Notify observers of configuration change
            for observer in self.observers:
                try:
                    if asyncio.iscoroutinefunction(observer):
                        await observer(old_config, self.config)
                    else:
                        observer(old_config, self.config)
                except Exception as e:
                    logger.error(f"Error notifying configuration observer: {e}")
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
    
    def add_observer(self, observer: callable):
        """Add configuration change observer"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: callable):
        """Remove configuration change observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def get_construction_rules(self) -> Dict[str, Any]:
        """Get construction industry business rules"""
        if not self.config:
            return {}
        return self.config.construction_rules.__dict__
    
    def update_construction_rules(self, rule_type: str, rules: Dict[str, Any]):
        """Update construction rules at runtime"""
        if not self.config:
            return
        
        if hasattr(self.config.construction_rules, rule_type):
            setattr(self.config.construction_rules, rule_type, rules)
            logger.info(f"Updated construction rules: {rule_type}")

# Global configuration manager instance
config_manager = ConfigurationManager()

async def get_config() -> AdvancedRAGConfig:
    """Get the current configuration, loading if necessary"""
    if config_manager.config is None:
        await config_manager.load_config()
    return config_manager.config

# Convenience functions for common configuration access
async def get_embedding_config() -> EmbeddingConfig:
    config = await get_config()
    return config.embedding

async def get_database_config() -> DatabaseConfig:
    config = await get_config()
    return config.database

async def get_llm_config() -> LLMConfig:
    config = await get_config()
    return config.llm

async def get_search_config() -> SearchConfig:
    config = await get_config()
    return config.search

async def get_construction_rules() -> ConstructionRulesConfig:
    config = await get_config()
    return config.construction_rules