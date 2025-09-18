import os
import json
from typing import Dict, List, Any, Optional
import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for database connections
firebase_db = None
chroma_client = None
chroma_collection = None

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global firebase_db
    
    if firebase_db is not None:
        print("Firebase already initialized")
        return firebase_db
    
    try:
        # Create credentials from environment variables
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        }
        
        # Initialize Firebase Admin
        cred = credentials.Certificate(firebase_config)
        
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        firebase_db = firestore.client()
        print("✅ Firebase initialized successfully")
        return firebase_db
        
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        raise

def initialize_chromadb():
    """Initialize ChromaDB with persistence"""
    global chroma_client, chroma_collection
    
    if chroma_collection is not None:
        print("ChromaDB already initialized")
        return chroma_collection
    
    try:
        persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_storage")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "construction_rag")
        
        # Create OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Initialize ChromaDB with persistence
        chroma_client = chromadb.PersistentClient(path=persist_path)
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        
        print(f"✅ ChromaDB initialized successfully")
        print(f"📊 Current collection size: {chroma_collection.count()}")
        return chroma_collection
        
    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        raise

def get_firebase_db():
    """Get Firebase database instance"""
    if firebase_db is None:
        initialize_firebase()
    return firebase_db

def get_chroma_collection():
    """Get ChromaDB collection instance"""
    if chroma_collection is None:
        initialize_chromadb()
    return chroma_collection

async def fetch_job_consumed_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job"""
    try:
        db = get_firebase_db()
        doc_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id).collection('data').document('consumed')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            data['company_id'] = company_id
            data['job_id'] = job_id
            data['data_type'] = 'consumed'
            return data
        else:
            print(f"No consumed data found for company {company_id}, job {job_id}")
            return None
            
    except Exception as e:
        print(f"Error fetching consumed job data: {e}")
        return None

async def fetch_job_complete_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch complete job data including estimate and schedule"""
    try:
        db = get_firebase_db()
        job_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            print(f"No job found for company {company_id}, job {job_id}")
            return None
        
        job_data = job_doc.to_dict()
        job_data['company_id'] = company_id
        job_data['job_id'] = job_id
        
        # Also get consumed data if it exists
        consumed_data = await fetch_job_consumed_data(company_id, job_id)
        if consumed_data:
            job_data['consumed_data'] = consumed_data
        
        return job_data
            
    except Exception as e:
        print(f"Error fetching complete job data: {e}")
        return None

def extract_estimate_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract estimate data from complete job data"""
    if not job_data or 'estimate' not in job_data:
        return None
    
    estimate_entries = job_data['estimate']
    if not estimate_entries or not isinstance(estimate_entries, list):
        return None
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'estimate',
        'entries': estimate_entries,
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
    }

def extract_schedule_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract schedule data from complete job data"""
    if not job_data or 'schedule' not in job_data:
        return None
    
    schedule_entries = job_data['schedule']
    if not schedule_entries or not isinstance(schedule_entries, list):
        return None
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'schedule',
        'entries': schedule_entries,
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', '),
        'schedule_last_updated': job_data.get('scheduleLastUpdated', '')
    }

async def fetch_all_job_complete_data(company_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all complete job data including consumed, estimate, and schedule"""
    try:
        db = get_firebase_db()
        all_jobs_data = []
        
        # If company_id is specified, only fetch that company's data
        if company_id:
            companies_to_process = [company_id]
        else:
            # Get all companies
            companies_ref = db.collection('companies')
            companies = companies_ref.stream()
            companies_to_process = [company.id for company in companies]
        
        for comp_id in companies_to_process:
            print(f"Processing company: {comp_id}")
            
            # Get all jobs for this company
            jobs_ref = db.collection('companies').document(comp_id).collection('jobs')
            jobs = jobs_ref.stream()
            
            for job in jobs:
                job_id = job.id
                job_data = job.to_dict()
                job_data['company_id'] = comp_id
                job_data['job_id'] = job_id
                
                # Get consumed data
                consumed_data = await fetch_job_consumed_data(comp_id, job_id)
                
                # Create separate data objects for each type
                job_datasets = []
                
                # Add consumed data if exists
                if consumed_data and 'entries' in consumed_data:
                    job_datasets.append(consumed_data)
                
                # Add estimate data if exists
                estimate_data = extract_estimate_data(job_data)
                if estimate_data:
                    job_datasets.append(estimate_data)
                
                # Add schedule data if exists
                schedule_data = extract_schedule_data(job_data)
                if schedule_data:
                    job_datasets.append(schedule_data)
                
                # Add all datasets for this job
                all_jobs_data.extend(job_datasets)
        
        print(f"✅ Fetched data for {len(all_jobs_data)} job datasets")
        return all_jobs_data
        
    except Exception as e:
        print(f"❌ Error fetching all job complete data: {e}")
        return []

# Keep the old function for backward compatibility
async def fetch_job_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job (backward compatibility)"""
    return await fetch_job_consumed_data(company_id, job_id)

async def fetch_all_job_data(company_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all job consumed data (backward compatibility)"""
    try:
        all_complete_data = await fetch_all_job_complete_data(company_id)
        # Filter to only return consumed data for backward compatibility
        consumed_data = [data for data in all_complete_data if data.get('data_type') == 'consumed']
        return consumed_data
    except Exception as e:
        print(f"❌ Error fetching all job data: {e}")
        return []

def clear_chroma_collection():
    """Clear all documents from ChromaDB collection (for testing)"""
    try:
        collection = get_chroma_collection()
        
        # Get all IDs and delete them
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
            print(f"🗑️  Cleared {len(all_data['ids'])} documents from collection")
        else:
            print("Collection is already empty")
            
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        raise

def test_connections():
    """Test all database connections"""
    results = {
        "firebase": False,
        "chromadb": False,
        "openai": False
    }
    
    # Test Firebase
    try:
        db = get_firebase_db()
        # Try to read from companies collection
        companies = db.collection('companies').limit(1).get()
        results["firebase"] = True
        print("✅ Firebase connection successful")
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
    
    # Test ChromaDB
    try:
        collection = get_chroma_collection()
        count = collection.count()
        results["chromadb"] = True
        print(f"✅ ChromaDB connection successful (documents: {count})")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
    
    # Test OpenAI (indirectly through ChromaDB embedding function)
    try:
        collection = get_chroma_collection()
        # Try to add a test embedding (we'll remove it immediately)
        test_id = "test_connection_id"
        collection.add(
            documents=["test connection"],
            ids=[test_id]
        )
        # Remove the test document
        collection.delete(ids=[test_id])
        results["openai"] = True
        print("✅ OpenAI API connection successful")
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
    
    return results

# Initialize connections when module is imported
def init_all():
    """Initialize all database connections"""
    try:
        initialize_firebase()
        initialize_chromadb()
        print("🚀 All database connections initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize databases: {e}")
        return False