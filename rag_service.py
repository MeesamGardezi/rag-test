import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime
import json

from app.database import get_chroma_collection, fetch_all_job_data, get_firebase_db
from app.embedding_service import EmbeddingService
from app.models import DocumentSource

class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = get_chroma_collection()
        
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process all Firebase job data and create embeddings"""
        print("🔄 Starting Firebase data processing...")
        
        start_time = datetime.now()
        stats = {
            'total_jobs_processed': 0,
            'total_entries_embedded': 0,
            'companies_processed': [],
            'processing_time_seconds': 0.0,
            'errors': []
        }
        
        try:
            # Fetch job data from Firebase
            job_data_list = await fetch_all_job_data(company_id)
            
            if not job_data_list:
                print("No job data found in Firebase")
                return stats
            
            # Process each job
            documents_to_add = []
            embeddings_to_add = []
            ids_to_add = []
            metadatas_to_add = []
            
            for job_data in job_data_list:
                try:
                    # Create text representation
                    text_content = self.embedding_service.create_job_text_representation(job_data)
                    
                    # Generate embedding
                    embedding = self.embedding_service.generate_embedding(text_content)
                    
                    # Create metadata
                    metadata = self.embedding_service.create_metadata(job_data)
                    
                    # Create unique ID
                    doc_id = f"{job_data['company_id']}_{job_data['job_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Add to batch
                    documents_to_add.append(text_content)
                    embeddings_to_add.append(embedding)
                    ids_to_add.append(doc_id)
                    metadatas_to_add.append(metadata)
                    
                    stats['total_jobs_processed'] += 1
                    stats['total_entries_embedded'] += len(job_data.get('entries', []))
                    
                    if job_data['company_id'] not in stats['companies_processed']:
                        stats['companies_processed'].append(job_data['company_id'])
                    
                except Exception as e:
                    error_msg = f"Error processing job {job_data.get('job_id', 'unknown')}: {str(e)}"
                    print(error_msg)
                    stats['errors'].append(error_msg)
            
            # Add all documents to ChromaDB in one batch
            if documents_to_add:
                print(f"📝 Adding {len(documents_to_add)} documents to ChromaDB...")
                
                self.collection.add(
                    documents=documents_to_add,
                    embeddings=embeddings_to_add,
                    ids=ids_to_add,
                    metadatas=metadatas_to_add
                )
                
                print(f"✅ Successfully added {len(documents_to_add)} documents")
            
            # Calculate processing time
            end_time = datetime.now()
            stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            print(f"✅ Processing complete! Processed {stats['total_jobs_processed']} jobs in {stats['processing_time_seconds']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Fatal error in process_firebase_data: {str(e)}"
            print(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Manually add a document to the collection"""
        try:
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text)
            
            # Create unique ID
            doc_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Add metadata timestamp
            metadata['added_at'] = datetime.now().isoformat()
            metadata['document_type'] = metadata.get('document_type', 'manual')
            
            # Add to collection
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata]
            )
            
            print(f"✅ Manually added document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"Error adding document: {e}")
            raise
    
    async def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            print(f"🔍 Querying: {question}")
            
            # Search for relevant documents
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    "answer": "I couldn't find any relevant information in the construction data.",
                    "sources": [],
                    "chunks": []
                }
            
            # Extract relevant chunks and metadata
            relevant_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            print(f"📊 Found {len(relevant_chunks)} relevant chunks")
            
            # Create sources list
            sources = []
            for metadata, distance in zip(metadatas, distances):
                source = DocumentSource(
                    job_name=metadata.get('job_name', 'Unknown'),
                    company_id=metadata.get('company_id', ''),
                    job_id=metadata.get('job_id', ''),
                    cost_code=str(metadata.get('cost_codes', [])[0]) if metadata.get('cost_codes') else None,
                    amount=f"${metadata.get('total_cost', 0):,.2f}" if metadata.get('total_cost') else None,
                    last_updated=metadata.get('last_updated', '')
                )
                sources.append(source)
            
            # Generate answer using OpenAI
            answer = await self._generate_answer(question, relevant_chunks)
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks": relevant_chunks
            }
            
        except Exception as e:
            print(f"Error in query: {e}")
            raise
    
    async def _generate_answer(self, question: str, relevant_chunks: List[str]) -> str:
        """Generate answer using OpenAI with retrieved context"""
        try:
            # Create context from relevant chunks
            context = "\n\n---\n\n".join(relevant_chunks)
            
            # Create prompt
            prompt = f"""You are a construction project assistant. Use the following cost and project information to answer the user's question. Be specific about costs, job names, and cost codes when available.

Context from construction projects:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Include specific dollar amounts, job names, and cost codes when relevant
- If you can't find the information in the context, say so clearly
- Be concise but informative
- Format costs with proper currency symbols and commas

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful construction project assistant that answers questions based on project cost data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
    async def get_available_jobs(self) -> List[str]:
        """Get list of available jobs in the system"""
        try:
            # Query all documents to get unique job names
            all_data = self.collection.get()
            
            job_names = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    job_name = metadata.get('job_name')
                    if job_name and job_name != 'Unknown':
                        job_names.add(job_name)
            
            return sorted(list(job_names))
            
        except Exception as e:
            print(f"Error getting available jobs: {e}")
            return []
    
    async def get_available_cost_codes(self) -> List[str]:
        """Get list of available cost codes in the system"""
        try:
            all_data = self.collection.get()
            
            cost_codes = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    codes = metadata.get('cost_codes', [])
                    for code in codes:
                        if code and code.strip():
                            cost_codes.add(code)
            
            return sorted(list(cost_codes))
            
        except Exception as e:
            print(f"Error getting available cost codes: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand data structure
            sample_data = self.collection.peek(limit=5)
            sample_metadata = sample_data.get('metadatas', []) if sample_data else []
            
            # Calculate some basic stats
            job_names = set()
            companies = set()
            total_cost = 0.0
            
            if sample_metadata:
                for metadata in sample_metadata:
                    job_name = metadata.get('job_name')
                    if job_name:
                        job_names.add(job_name)
                    
                    company_id = metadata.get('company_id') 
                    if company_id:
                        companies.add(company_id)
                    
                    cost = metadata.get('total_cost', 0)
                    if isinstance(cost, (int, float)):
                        total_cost += cost
            
            return {
                'total_documents': count,
                'unique_jobs_sample': len(job_names),
                'unique_companies_sample': len(companies),
                'sample_total_cost': total_cost,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_old_documents(self, days_old: int = 7) -> int:
        """Clear documents older than specified days"""
        try:
            # This is a simple implementation - in production you'd want more sophisticated cleanup
            all_data = self.collection.get()
            
            if not all_data['ids']:
                return 0
            
            # For now, just return count - actual cleanup would need timestamp tracking
            return len(all_data['ids'])
            
        except Exception as e:
            print(f"Error clearing old documents: {e}")
            return 0