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

async def fetch_job_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job"""
    try:
        db = get_firebase_db()
        doc_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id).collection('data').document('consumed')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            data['company_id'] = company_id
            data['job_id'] = job_id
            return data
        else:
            print(f"No data found for company {company_id}, job {job_id}")
            return None
            
    except Exception as e:
        print(f"Error fetching job data: {e}")
        return None

async def fetch_all_job_data(company_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all job consumed data"""
    try:
        db = get_firebase_db()
        all_jobs = []
        
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
                
                # Try to get consumed data
                job_data = await fetch_job_data(comp_id, job_id)
                if job_data and 'entries' in job_data:
                    all_jobs.append(job_data)
        
        print(f"✅ Fetched data for {len(all_jobs)} jobs")
        return all_jobs
        
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