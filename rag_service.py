import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime
import json

from database import get_chroma_collection, fetch_all_job_complete_data, get_firebase_db
from embedding_service import EmbeddingService
from models import DocumentSource

class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = get_chroma_collection()
        
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process all Firebase job data (consumed, estimate, schedule) and create embeddings"""
        print("🔄 Starting comprehensive Firebase data processing...")
        
        start_time = datetime.now()
        stats = {
            'total_jobs_processed': 0,
            'total_datasets_embedded': 0,
            'consumed_datasets': 0,
            'estimate_datasets': 0,
            'schedule_datasets': 0,
            'companies_processed': [],
            'processing_time_seconds': 0.0,
            'errors': []
        }
        
        try:
            # Fetch all job data from Firebase
            job_datasets = await fetch_all_job_complete_data(company_id)
            
            if not job_datasets:
                print("No job data found in Firebase")
                return stats
            
            print(f"📊 Found {len(job_datasets)} datasets to process")
            
            # Process each dataset
            documents_to_add = []
            embeddings_to_add = []
            ids_to_add = []
            metadatas_to_add = []
            
            jobs_processed = set()
            
            for job_dataset in job_datasets:
                try:
                    data_type = job_dataset.get('data_type', 'unknown')
                    job_id = job_dataset.get('job_id', 'unknown')
                    company_id_current = job_dataset.get('company_id', 'unknown')
                    
                    # Create text representation
                    text_content = self.embedding_service.create_job_text_representation(job_dataset)
                    
                    # Generate embedding
                    embedding = self.embedding_service.generate_embedding(text_content)
                    
                    # Create metadata
                    metadata = self.embedding_service.create_metadata(job_dataset)
                    
                    # Create unique ID that includes data type
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    doc_id = f"{company_id_current}_{job_id}_{data_type}_{timestamp}"
                    
                    # Add to batch
                    documents_to_add.append(text_content)
                    embeddings_to_add.append(embedding)
                    ids_to_add.append(doc_id)
                    metadatas_to_add.append(metadata)
                    
                    # Update stats
                    stats['total_datasets_embedded'] += 1
                    jobs_processed.add(f"{company_id_current}_{job_id}")
                    
                    if data_type == 'consumed':
                        stats['consumed_datasets'] += 1
                    elif data_type == 'estimate':
                        stats['estimate_datasets'] += 1
                    elif data_type == 'schedule':
                        stats['schedule_datasets'] += 1
                    
                    if company_id_current not in stats['companies_processed']:
                        stats['companies_processed'].append(company_id_current)
                    
                    print(f"✅ Processed {data_type} data for job {job_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing {job_dataset.get('data_type', 'unknown')} data for job {job_dataset.get('job_id', 'unknown')}: {str(e)}"
                    print(error_msg)
                    stats['errors'].append(error_msg)
            
            # Update job count
            stats['total_jobs_processed'] = len(jobs_processed)
            
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
                print(f"   - {stats['consumed_datasets']} consumed datasets")
                print(f"   - {stats['estimate_datasets']} estimate datasets")
                print(f"   - {stats['schedule_datasets']} schedule datasets")
            
            # Calculate processing time
            end_time = datetime.now()
            stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            print(f"✅ Processing complete! Processed {stats['total_jobs_processed']} jobs ({stats['total_datasets_embedded']} datasets) in {stats['processing_time_seconds']:.2f} seconds")
            
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
    
    async def query(self, question: str, n_results: int = 5, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the RAG system with optional data type filtering"""
        try:
            print(f"🔍 Querying: {question}")
            if data_types:
                print(f"📋 Filtering by data types: {data_types}")
            
            # Search for relevant documents
            query_filter = None
            if data_types:
                # Create filter for specific data types
                query_filter = {"data_type": {"$in": data_types}}
            
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                where=query_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    "answer": "I couldn't find any relevant information in the construction data.",
                    "sources": [],
                    "chunks": [],
                    "data_types_found": []
                }
            
            # Extract relevant chunks and metadata
            relevant_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            print(f"📊 Found {len(relevant_chunks)} relevant chunks")
            
            # Analyze data types in results
            data_types_found = set()
            document_types_found = set()
            
            # Create sources list with enhanced information
            sources = []
            for metadata, distance in zip(metadatas, distances):
                data_type = metadata.get('data_type', 'unknown')
                data_types_found.add(data_type)
                document_types_found.add(metadata.get('document_type', 'unknown'))
                
                # Create cost information based on data type
                cost_info = None
                if data_type == 'consumed':
                    cost_info = f"${metadata.get('total_cost', 0):,.2f} consumed"
                elif data_type == 'estimate':
                    estimated = metadata.get('total_estimated_cost', 0)
                    budgeted = metadata.get('total_budgeted_cost', 0)
                    cost_info = f"${estimated:,.2f} est. / ${budgeted:,.2f} budgeted"
                elif data_type == 'schedule':
                    hours = metadata.get('total_planned_hours', 0)
                    consumed = metadata.get('total_consumed_hours', 0)
                    cost_info = f"{hours:.1f}h planned / {consumed:.1f}h consumed"
                
                source = DocumentSource(
                    job_name=metadata.get('job_name', 'Unknown'),
                    company_id=metadata.get('company_id', ''),
                    job_id=metadata.get('job_id', ''),
                    cost_code=metadata.get('categories', '') or metadata.get('areas', ''),
                    amount=cost_info,
                    last_updated=metadata.get('last_updated', '')
                )
                sources.append(source)
            
            # Generate answer using OpenAI with enhanced context
            answer = await self._generate_enhanced_answer(question, relevant_chunks, metadatas)
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks": relevant_chunks,
                "data_types_found": list(data_types_found),
                "document_types_found": list(document_types_found)
            }
            
        except Exception as e:
            print(f"Error in query: {e}")
            raise
    
    async def _generate_enhanced_answer(self, question: str, relevant_chunks: List[str], metadatas: List[Dict]) -> str:
        """Generate answer using OpenAI with enhanced context awareness"""
        try:
            # Create context from relevant chunks with data type information
            context_parts = []
            
            for chunk, metadata in zip(relevant_chunks, metadatas):
                data_type = metadata.get('data_type', 'unknown')
                job_name = metadata.get('job_name', 'Unknown Job')
                
                context_parts.append(f"[{data_type.upper()} DATA - {job_name}]\n{chunk}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create enhanced prompt
            prompt = f"""You are a construction project assistant with access to comprehensive project data including consumed costs, estimates, and schedules. Use the following information to answer the user's question accurately and comprehensively.

Context from construction projects:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Distinguish between different data types (CONSUMED, ESTIMATE, SCHEDULE) when relevant
- Include specific dollar amounts, hours, dates, job names, and other details when available
- If comparing data types (e.g., estimated vs consumed), highlight the differences
- For schedule questions, include timeline information and task details
- If you can't find the information in the context, say so clearly
- Be concise but informative
- Format costs with proper currency symbols and commas

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful construction project assistant that answers questions based on comprehensive project data including costs, estimates, and schedules."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
    async def get_available_jobs(self) -> List[Dict[str, Any]]:
        """Get list of available jobs in the system with data type information"""
        try:
            # Query all documents to get unique job names
            all_data = self.collection.get()
            
            jobs_info = {}
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    job_name = metadata.get('job_name')
                    job_id = metadata.get('job_id')
                    data_type = metadata.get('data_type', 'unknown')
                    
                    if job_name and job_name != 'Unknown':
                        job_key = f"{job_name}_{job_id}"
                        if job_key not in jobs_info:
                            jobs_info[job_key] = {
                                'job_name': job_name,
                                'job_id': job_id,
                                'data_types': [],
                                'company_id': metadata.get('company_id', '')
                            }
                        
                        if data_type not in jobs_info[job_key]['data_types']:
                            jobs_info[job_key]['data_types'].append(data_type)
            
            return list(jobs_info.values())
            
        except Exception as e:
            print(f"Error getting available jobs: {e}")
            return []
    
    async def get_data_types_summary(self) -> Dict[str, Any]:
        """Get summary of available data types in the system"""
        try:
            all_data = self.collection.get()
            
            summary = {
                'consumed': 0,
                'estimate': 0,
                'schedule': 0,
                'unknown': 0,
                'total_documents': 0
            }
            
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    data_type = metadata.get('data_type', 'unknown')
                    if data_type in summary:
                        summary[data_type] += 1
                    else:
                        summary['unknown'] += 1
                    summary['total_documents'] += 1
            
            return summary
            
        except Exception as e:
            print(f"Error getting data types summary: {e}")
            return {'error': str(e)}
    
    async def get_available_cost_codes(self) -> List[str]:
        """Get list of available cost codes in the system"""
        try:
            all_data = self.collection.get()
            
            cost_codes = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    # Get cost codes from consumed data
                    codes = metadata.get('cost_codes', '')
                    if codes:
                        for code in codes.split(', '):
                            if code and code.strip():
                                cost_codes.add(code.strip())
                    
                    # Get areas from estimate data
                    areas = metadata.get('areas', '')
                    if areas:
                        for area in areas.split(', '):
                            if area and area.strip():
                                cost_codes.add(f"Area: {area.strip()}")
            
            return sorted(list(cost_codes))
            
        except Exception as e:
            print(f"Error getting available cost codes: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand data structure
            sample_data = self.collection.peek(limit=20)
            sample_metadata = sample_data.get('metadatas', []) if sample_data else []
            
            # Calculate stats by data type
            stats_by_type = {
                'consumed': {'count': 0, 'total_cost': 0.0, 'jobs': set()},
                'estimate': {'count': 0, 'total_estimated': 0.0, 'total_budgeted': 0.0, 'jobs': set()},
                'schedule': {'count': 0, 'total_hours': 0.0, 'total_consumed': 0.0, 'jobs': set()},
                'unknown': {'count': 0, 'jobs': set()}
            }
            
            companies = set()
            
            if sample_metadata:
                for metadata in sample_metadata:
                    data_type = metadata.get('data_type', 'unknown')
                    job_id = metadata.get('job_id')
                    company_id = metadata.get('company_id')
                    
                    if company_id:
                        companies.add(company_id)
                    
                    if data_type in stats_by_type:
                        stats_by_type[data_type]['count'] += 1
                        if job_id:
                            stats_by_type[data_type]['jobs'].add(job_id)
                        
                        if data_type == 'consumed':
                            cost = metadata.get('total_cost', 0)
                            if isinstance(cost, (int, float)):
                                stats_by_type[data_type]['total_cost'] += cost
                        
                        elif data_type == 'estimate':
                            estimated = metadata.get('total_estimated_cost', 0)
                            budgeted = metadata.get('total_budgeted_cost', 0)
                            if isinstance(estimated, (int, float)):
                                stats_by_type[data_type]['total_estimated'] += estimated
                            if isinstance(budgeted, (int, float)):
                                stats_by_type[data_type]['total_budgeted'] += budgeted
                        
                        elif data_type == 'schedule':
                            hours = metadata.get('total_planned_hours', 0)
                            consumed = metadata.get('total_consumed_hours', 0)
                            if isinstance(hours, (int, float)):
                                stats_by_type[data_type]['total_hours'] += hours
                            if isinstance(consumed, (int, float)):
                                stats_by_type[data_type]['total_consumed'] += consumed
                    else:
                        stats_by_type['unknown']['count'] += 1
                        if job_id:
                            stats_by_type['unknown']['jobs'].add(job_id)
            
            # Convert sets to counts
            for type_stats in stats_by_type.values():
                type_stats['unique_jobs'] = len(type_stats['jobs'])
                del type_stats['jobs']  # Remove the set to make it JSON serializable
            
            return {
                'total_documents': count,
                'unique_companies': len(companies),
                'stats_by_type': stats_by_type,
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