import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from datetime import datetime
import json
import re

from database import get_chroma_collection, fetch_all_job_complete_data, get_firebase_db
from embedding_service import EmbeddingService
from models import DocumentSource

class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = get_chroma_collection()
        
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all Firebase job data with multi-granularity embeddings:
        - Job-level summaries
        - Individual row-level details for estimates
        """
        print("🔄 Starting comprehensive Firebase data processing with row-level granularity...")
        
        start_time = datetime.now()
        stats = {
            'total_jobs_processed': 0,
            'total_datasets_embedded': 0,
            'total_rows_embedded': 0,
            'consumed_datasets': 0,
            'estimate_datasets': 0,
            'estimate_rows': 0,
            'flooring_estimate_datasets': 0,
            'flooring_estimate_rows': 0,
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
            
            # Process each dataset with multi-granularity approach
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
                    
                    # Track job processing
                    jobs_processed.add(f"{company_id_current}_{job_id}")
                    
                    if company_id_current not in stats['companies_processed']:
                        stats['companies_processed'].append(company_id_current)
                    
                    # Process based on data type
                    if data_type == 'estimate':
                        # For estimates, create both job-level and row-level embeddings
                        await self._process_estimate_multi_granularity(
                            job_dataset, documents_to_add, embeddings_to_add,
                            ids_to_add, metadatas_to_add, stats
                        )
                    
                    elif data_type == 'flooring_estimate':
                        # For flooring estimates, create both job-level and row-level embeddings
                        await self._process_flooring_estimate_multi_granularity(
                            job_dataset, documents_to_add, embeddings_to_add,
                            ids_to_add, metadatas_to_add, stats
                        )
                    
                    elif data_type in ['consumed', 'schedule']:
                        # For consumed and schedule, use existing approach (job-level)
                        text_content = self.embedding_service.create_job_text_representation(job_dataset)
                        embedding = self.embedding_service.generate_embedding(text_content)
                        metadata = self.embedding_service.create_metadata(job_dataset)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        doc_id = f"{company_id_current}_{job_id}_{data_type}_{timestamp}"
                        
                        documents_to_add.append(text_content)
                        embeddings_to_add.append(embedding)
                        ids_to_add.append(doc_id)
                        metadatas_to_add.append(metadata)
                        
                        stats['total_datasets_embedded'] += 1
                        if data_type == 'consumed':
                            stats['consumed_datasets'] += 1
                        elif data_type == 'schedule':
                            stats['schedule_datasets'] += 1
                        
                        print(f"✅ Processed {data_type} data for job {job_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing {job_dataset.get('data_type', 'unknown')} data for job {job_dataset.get('job_id', 'unknown')}: {str(e)}"
                    print(error_msg)
                    stats['errors'].append(error_msg)
            
            # Update job count
            stats['total_jobs_processed'] = len(jobs_processed)
            
            # Add all documents to ChromaDB in batches
            if documents_to_add:
                print(f"📝 Adding {len(documents_to_add)} documents to ChromaDB...")
                
                # Process in batches of 100 to avoid overwhelming ChromaDB
                batch_size = 100
                for i in range(0, len(documents_to_add), batch_size):
                    end_idx = min(i + batch_size, len(documents_to_add))
                    
                    self.collection.add(
                        documents=documents_to_add[i:end_idx],
                        embeddings=embeddings_to_add[i:end_idx],
                        ids=ids_to_add[i:end_idx],
                        metadatas=metadatas_to_add[i:end_idx]
                    )
                    
                    print(f"  Batch {i//batch_size + 1}/{(len(documents_to_add) + batch_size - 1)//batch_size} added")
                
                print(f"✅ Successfully added {len(documents_to_add)} documents")
                print(f"   - {stats['consumed_datasets']} consumed datasets")
                print(f"   - {stats['estimate_datasets']} estimate summaries")
                print(f"   - {stats['estimate_rows']} estimate rows")
                print(f"   - {stats['flooring_estimate_datasets']} flooring estimate summaries")
                print(f"   - {stats['flooring_estimate_rows']} flooring estimate rows")
                print(f"   - {stats['schedule_datasets']} schedule datasets")
            
            # Calculate processing time
            end_time = datetime.now()
            stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            print(f"✅ Processing complete! Processed {stats['total_jobs_processed']} jobs")
            print(f"   Total documents: {stats['total_datasets_embedded']}")
            print(f"   Total rows: {stats['total_rows_embedded']}")
            print(f"   Time: {stats['processing_time_seconds']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Fatal error in process_firebase_data: {str(e)}"
            print(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    async def _process_estimate_multi_granularity(
        self, job_dataset: Dict[str, Any],
        documents: List[str], embeddings: List[List[float]],
        ids: List[str], metadatas: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        """Process estimate data with both job-level and row-level embeddings"""
        job_id = job_dataset.get('job_id', 'unknown')
        company_id = job_dataset.get('company_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Create job-level summary embedding
        summary_text = self.embedding_service.create_job_text_representation(job_dataset)
        summary_embedding = self.embedding_service.generate_embedding(summary_text)
        summary_metadata = self.embedding_service.create_metadata(job_dataset)
        
        summary_id = f"{company_id}_{job_id}_estimate_summary_{timestamp}"
        documents.append(summary_text)
        embeddings.append(summary_embedding)
        ids.append(summary_id)
        metadatas.append(summary_metadata)
        
        stats['estimate_datasets'] += 1
        stats['total_datasets_embedded'] += 1
        
        # 2. Create individual row-level embeddings
        entries = job_dataset.get('entries', [])
        job_context = {
            'job_name': job_dataset.get('job_name', 'Unknown'),
            'company_id': company_id,
            'job_id': job_id,
            'client_name': job_dataset.get('client_name', ''),
            'site_location': job_dataset.get('site_location', ''),
            'last_updated': job_dataset.get('last_updated', ''),
            'estimate_type': job_dataset.get('estimate_type', 'general')
        }
        
        for entry in entries:
            row_num = entry.get('row_number', 0)
            
            # Create row-specific text and embedding
            row_text = self.embedding_service.create_estimate_row_text(entry, job_context)
            row_embedding = self.embedding_service.generate_embedding(row_text)
            row_metadata = self.embedding_service.create_estimate_row_metadata(entry, job_context)
            
            row_id = f"{company_id}_{job_id}_estimate_row_{row_num}_{timestamp}"
            documents.append(row_text)
            embeddings.append(row_embedding)
            ids.append(row_id)
            metadatas.append(row_metadata)
            
            stats['estimate_rows'] += 1
            stats['total_rows_embedded'] += 1
        
        print(f"✅ Processed estimate for job {job_id}: 1 summary + {len(entries)} rows")
    
    async def _process_flooring_estimate_multi_granularity(
        self, job_dataset: Dict[str, Any],
        documents: List[str], embeddings: List[List[float]],
        ids: List[str], metadatas: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        """Process flooring estimate data with both job-level and row-level embeddings"""
        job_id = job_dataset.get('job_id', 'unknown')
        company_id = job_dataset.get('company_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Create job-level summary embedding
        summary_text = self.embedding_service.create_job_text_representation(job_dataset)
        summary_embedding = self.embedding_service.generate_embedding(summary_text)
        summary_metadata = self.embedding_service.create_metadata(job_dataset)
        
        summary_id = f"{company_id}_{job_id}_flooring_summary_{timestamp}"
        documents.append(summary_text)
        embeddings.append(summary_embedding)
        ids.append(summary_id)
        metadatas.append(summary_metadata)
        
        stats['flooring_estimate_datasets'] += 1
        stats['total_datasets_embedded'] += 1
        
        # 2. Create individual row-level embeddings
        entries = job_dataset.get('entries', [])
        job_context = {
            'job_name': job_dataset.get('job_name', 'Unknown'),
            'company_id': company_id,
            'job_id': job_id,
            'client_name': job_dataset.get('client_name', ''),
            'site_location': job_dataset.get('site_location', '')
        }
        
        for entry in entries:
            row_num = entry.get('row_number', 0)
            
            # Create row-specific text and embedding
            row_text = self.embedding_service.create_flooring_estimate_row_text(entry, job_context)
            row_embedding = self.embedding_service.generate_embedding(row_text)
            row_metadata = self.embedding_service.create_flooring_estimate_row_metadata(entry, job_context)
            
            row_id = f"{company_id}_{job_id}_flooring_row_{row_num}_{timestamp}"
            documents.append(row_text)
            embeddings.append(row_embedding)
            ids.append(row_id)
            metadatas.append(row_metadata)
            
            stats['flooring_estimate_rows'] += 1
            stats['total_rows_embedded'] += 1
        
        print(f"✅ Processed flooring estimate for job {job_id}: 1 summary + {len(entries)} rows")
    
    def _detect_row_specific_query(self, question: str) -> Optional[int]:
        """Detect if query is asking about a specific row number"""
        # Pattern matching for row numbers
        patterns = [
            r'row\s+#?(\d+)',
            r'line\s+#?(\d+)',
            r'item\s+#?(\d+)',
            r'entry\s+#?(\d+)',
            r'(\d+)(?:st|nd|rd|th)\s+row',
            r'row\s+number\s+(\d+)',
        ]
        
        question_lower = question.lower()
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _detect_allowance_query(self, question: str) -> bool:
        """Detect if query is asking about allowances"""
        allowance_keywords = ['allowance', 'allowances', 'contingency', 'contingencies']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in allowance_keywords)
    
    def _detect_materials_query(self, question: str) -> bool:
        """Detect if query is asking about materials"""
        materials_keywords = ['material', 'materials', 'supplies', 'items needed']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in materials_keywords)
    
    async def query(self, question: str, n_results: int = 10, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhanced query with intelligent routing for:
        - Row-specific queries
        - Allowance queries
        - Materials queries
        - General estimate queries
        """
        try:
            print(f"🔍 Querying: {question}")
            if data_types:
                print(f"📋 Filtering by data types: {data_types}")
            
            # Detect query intent
            row_number = self._detect_row_specific_query(question)
            is_allowance_query = self._detect_allowance_query(question)
            is_materials_query = self._detect_materials_query(question)
            
            # Build filter based on query intent
            query_filter = self._build_query_filter(
                data_types=data_types,
                row_number=row_number,
                is_allowance=is_allowance_query,
                has_materials=is_materials_query
            )
            
            # Adjust n_results based on query type
            if row_number is not None:
                # For row-specific queries, get fewer results but focus on that row
                n_results = min(n_results, 5)
                print(f"🎯 Row-specific query detected: Row #{row_number}")
            elif is_allowance_query:
                print(f"💰 Allowance query detected")
            elif is_materials_query:
                print(f"🔧 Materials query detected")
            
            # Search for relevant documents
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
                    "data_types_found": [],
                    "row_numbers_found": []
                }
            
            # Extract relevant chunks and metadata
            relevant_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            print(f"📊 Found {len(relevant_chunks)} relevant chunks")
            
            # Analyze results
            data_types_found = set()
            document_types_found = set()
            row_numbers_found = set()
            granularities_found = set()
            
            # Create sources list with enhanced information
            sources = []
            for metadata, distance in zip(metadatas, distances):
                data_type = metadata.get('data_type', 'unknown')
                data_types_found.add(data_type)
                document_types_found.add(metadata.get('document_type', 'unknown'))
                granularities_found.add(metadata.get('granularity', 'unknown'))
                
                # Track row numbers if present
                if 'row_number' in metadata:
                    row_numbers_found.add(metadata['row_number'])
                
                # Create appropriate cost/value information based on data type
                cost_info = self._format_cost_info(metadata, data_type)
                
                source = DocumentSource(
                    job_name=metadata.get('job_name', 'Unknown'),
                    company_id=metadata.get('company_id', ''),
                    job_id=metadata.get('job_id', ''),
                    cost_code=self._get_display_identifier(metadata),
                    amount=cost_info,
                    last_updated=metadata.get('last_updated', '')
                )
                sources.append(source)
            
            # Generate enhanced answer
            answer = await self._generate_enhanced_answer(
                question, relevant_chunks, metadatas,
                row_number=row_number,
                is_allowance=is_allowance_query,
                is_materials=is_materials_query
            )
            
            return {
                "answer": answer,
                "sources": sources,
                "chunks": relevant_chunks,
                "data_types_found": list(data_types_found),
                "document_types_found": list(document_types_found),
                "granularities_found": list(granularities_found),
                "row_numbers_found": sorted(list(row_numbers_found)) if row_numbers_found else []
            }
            
        except Exception as e:
            print(f"Error in query: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_query_filter(
        self, 
        data_types: Optional[List[str]] = None,
        row_number: Optional[int] = None,
        is_allowance: bool = False,
        has_materials: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB query filter based on query intent"""
        filters = []
        
        # Data type filter
        if data_types:
            filters.append({"data_type": {"$in": data_types}})
        
        # Row number filter
        if row_number is not None:
            filters.append({"row_number": row_number})
        
        # Allowance filter
        if is_allowance:
            filters.append({"is_allowance": True})
        
        # Materials filter
        if has_materials:
            filters.append({"has_materials": True})
        
        # Combine filters
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def _get_display_identifier(self, metadata: Dict[str, Any]) -> str:
        """Get appropriate display identifier based on document type"""
        granularity = metadata.get('granularity', 'unknown')
        
        if granularity == 'row':
            # Row-level data
            row_num = metadata.get('row_number', '?')
            area = metadata.get('area', '')
            task = metadata.get('task_scope', '')
            cost_code = metadata.get('costCode', metadata.get('cost_code', ''))
            
            if area and task:
                return f"Row {row_num}: {area} - {task} ({cost_code})"
            elif cost_code:
                return f"Row {row_num}: {cost_code}"
            else:
                return f"Row {row_num}"
        else:
            # Job-level data
            categories = metadata.get('categories', '') or metadata.get('areas', '')
            return categories if categories else 'Summary'
    
    def _format_cost_info(self, metadata: Dict[str, Any], data_type: str) -> Optional[str]:
        """Format cost information based on data type and granularity"""
        granularity = metadata.get('granularity', 'job')
        
        if data_type == 'estimate' and granularity == 'row':
            # Estimate row
            total = metadata.get('total', 0)
            budgeted = metadata.get('budgeted_total', 0)
            variance = metadata.get('variance', 0)
            return f"${total:,.2f} est. / ${budgeted:,.2f} budgeted (Δ ${variance:,.2f})"
        
        elif data_type == 'estimate' and granularity == 'job':
            # Estimate summary
            estimated = metadata.get('total_estimated_cost', 0)
            budgeted = metadata.get('total_budgeted_cost', 0)
            return f"${estimated:,.2f} est. / ${budgeted:,.2f} budgeted"
        
        elif data_type == 'flooring_estimate' and granularity == 'row':
            # Flooring row
            cost = metadata.get('total_cost', 0)
            sale = metadata.get('sale_price', 0)
            profit = metadata.get('profit', 0)
            return f"Cost: ${cost:,.2f} | Sale: ${sale:,.2f} | Profit: ${profit:,.2f}"
        
        elif data_type == 'flooring_estimate' and granularity == 'job':
            # Flooring summary
            total_cost = metadata.get('total_cost', 0)
            total_sale = metadata.get('total_sale', 0)
            return f"Cost: ${total_cost:,.2f} | Sale: ${total_sale:,.2f}"
        
        elif data_type == 'consumed':
            total_cost = metadata.get('total_cost', 0)
            return f"${total_cost:,.2f} consumed"
        
        elif data_type == 'schedule':
            hours = metadata.get('total_planned_hours', 0)
            consumed = metadata.get('total_consumed_hours', 0)
            return f"{hours:.1f}h planned / {consumed:.1f}h consumed"
        
        return None
    
    async def _generate_enhanced_answer(
        self, question: str, relevant_chunks: List[str], metadatas: List[Dict],
        row_number: Optional[int] = None,
        is_allowance: bool = False,
        is_materials: bool = False
    ) -> str:
        """Generate answer using OpenAI with enhanced context awareness"""
        try:
            # Create context from relevant chunks with appropriate labeling
            context_parts = []
            
            for chunk, metadata in zip(relevant_chunks, metadatas):
                data_type = metadata.get('data_type', 'unknown').upper()
                granularity = metadata.get('granularity', 'unknown')
                job_name = metadata.get('job_name', 'Unknown Job')
                
                # Create descriptive label
                if granularity == 'row':
                    row_num = metadata.get('row_number', '?')
                    label = f"[{data_type} - ROW {row_num} - {job_name}]"
                else:
                    label = f"[{data_type} SUMMARY - {job_name}]"
                
                context_parts.append(f"{label}\n{chunk}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create specialized prompt based on query type
            if row_number is not None:
                special_instruction = f"The user is specifically asking about ROW #{row_number}. Focus your answer on that specific row's details including description, quantities, rates, and any notes/remarks."
            elif is_allowance:
                special_instruction = "The user is asking about allowances. Focus on items marked as allowances, their purposes, amounts, and any special conditions."
            elif is_materials:
                special_instruction = "The user is asking about materials. Focus on material specifications, quantities, suppliers, and costs."
            else:
                special_instruction = "Provide a comprehensive answer based on the available data."
            
            # Create enhanced prompt
            prompt = f"""You are a construction project assistant with access to detailed project data including estimate rows, schedules, and costs. Use the following information to answer the user's question accurately and comprehensively.

Context from construction projects:
{context}

Question: {question}

Instructions:
- {special_instruction}
- Answer based only on the provided context
- Distinguish between different data types (ESTIMATE, SCHEDULE, CONSUMED) when relevant
- Include specific details like row numbers, dollar amounts, hours, dates, job names, and descriptions when available
- When showing row-specific information, always include the row number
- For estimates, distinguish between estimated and budgeted amounts
- If comparing data, highlight the differences clearly
- For schedule questions, include timeline information and task details
- If the context contains row-level data, provide those specific details rather than generalizations
- If materials are mentioned in the context, list them with quantities and costs
- If you can't find the information in the context, say so clearly
- Be concise but informative
- Format costs with proper currency symbols and commas
- When listing multiple rows, organize them clearly with row numbers

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful construction project assistant that answers questions based on detailed project data including individual estimate rows, schedules, and costs. Always cite specific row numbers when discussing estimate line items."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
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
                    granularity = metadata.get('granularity', 'unknown')
                    
                    if job_name and job_name != 'Unknown':
                        job_key = f"{job_name}_{job_id}"
                        if job_key not in jobs_info:
                            jobs_info[job_key] = {
                                'job_name': job_name,
                                'job_id': job_id,
                                'data_types': set(),
                                'has_row_level_data': False,
                                'company_id': metadata.get('company_id', '')
                            }
                        
                        jobs_info[job_key]['data_types'].add(data_type)
                        
                        if granularity == 'row':
                            jobs_info[job_key]['has_row_level_data'] = True
            
            # Convert sets to lists
            result = []
            for job_info in jobs_info.values():
                job_info['data_types'] = list(job_info['data_types'])
                result.append(job_info)
            
            return result
            
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
                'flooring_estimate': 0,
                'schedule': 0,
                'unknown': 0,
                'total_documents': 0,
                'row_level_documents': 0,
                'job_level_documents': 0
            }
            
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    data_type = metadata.get('data_type', 'unknown')
                    granularity = metadata.get('granularity', 'unknown')
                    
                    if data_type in summary:
                        summary[data_type] += 1
                    else:
                        summary['unknown'] += 1
                    
                    if granularity == 'row':
                        summary['row_level_documents'] += 1
                    elif granularity == 'job':
                        summary['job_level_documents'] += 1
                    
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
                    # Get cost codes from various fields
                    code = metadata.get('cost_code', '') or metadata.get('costCode', '')
                    if code and code.strip():
                        cost_codes.add(code.strip())
                    
                    # Get from cost_codes field (consumed data)
                    codes = metadata.get('cost_codes', '')
                    if codes:
                        for code in codes.split(', '):
                            if code and code.strip():
                                cost_codes.add(code.strip())
                    
                    # Get areas from estimate data
                    area = metadata.get('area', '')
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
            sample_data = self.collection.peek(limit=100)
            sample_metadata = sample_data.get('metadatas', []) if sample_data else []
            
            # Calculate detailed stats
            stats_by_type = {
                'consumed': {'count': 0, 'total_cost': 0.0, 'jobs': set()},
                'estimate': {
                    'count': 0, 'total_estimated': 0.0, 'total_budgeted': 0.0,
                    'jobs': set(), 'row_count': 0, 'summary_count': 0
                },
                'flooring_estimate': {
                    'count': 0, 'total_cost': 0.0, 'total_sale': 0.0,
                    'jobs': set(), 'row_count': 0, 'summary_count': 0
                },
                'schedule': {'count': 0, 'total_hours': 0.0, 'total_consumed': 0.0, 'jobs': set()},
                'unknown': {'count': 0, 'jobs': set()}
            }
            
            companies = set()
            
            if sample_metadata:
                for metadata in sample_metadata:
                    data_type = metadata.get('data_type', 'unknown')
                    granularity = metadata.get('granularity', 'unknown')
                    job_id = metadata.get('job_id')
                    company_id = metadata.get('company_id')
                    
                    if company_id:
                        companies.add(company_id)
                    
                    if data_type in stats_by_type:
                        stats_by_type[data_type]['count'] += 1
                        if job_id:
                            stats_by_type[data_type]['jobs'].add(job_id)
                        
                        # Track granularity for estimates
                        if data_type in ['estimate', 'flooring_estimate']:
                            if granularity == 'row':
                                stats_by_type[data_type]['row_count'] += 1
                            elif granularity == 'job':
                                stats_by_type[data_type]['summary_count'] += 1
                        
                        # Aggregate costs
                        if data_type == 'consumed':
                            cost = metadata.get('total_cost', 0)
                            if isinstance(cost, (int, float)):
                                stats_by_type[data_type]['total_cost'] += cost
                        
                        elif data_type == 'estimate' and granularity == 'job':
                            estimated = metadata.get('total_estimated_cost', 0)
                            budgeted = metadata.get('total_budgeted_cost', 0)
                            if isinstance(estimated, (int, float)):
                                stats_by_type[data_type]['total_estimated'] += estimated
                            if isinstance(budgeted, (int, float)):
                                stats_by_type[data_type]['total_budgeted'] += budgeted
                        
                        elif data_type == 'flooring_estimate' and granularity == 'job':
                            cost = metadata.get('total_cost', 0)
                            sale = metadata.get('total_sale', 0)
                            if isinstance(cost, (int, float)):
                                stats_by_type[data_type]['total_cost'] += cost
                            if isinstance(sale, (int, float)):
                                stats_by_type[data_type]['total_sale'] += sale
                        
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
            import traceback
            traceback.print_exc()
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