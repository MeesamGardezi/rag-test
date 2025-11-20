import os
from typing import List, Dict, Any, Optional, Set, Tuple
from openai import OpenAI
from datetime import datetime
import json
import re
from difflib import SequenceMatcher

from database import get_chroma_collection, fetch_all_job_complete_data, get_firebase_db
from embedding_service import EmbeddingService
from models import DocumentSource

class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = get_chroma_collection()
        
        # Query preprocessing synonyms and expansions
        self.query_synonyms = {
            'cleanup': ['cleanup', 'clean up', 'clean-up', 'cleaning'],
            'electrical': ['electrical', 'electric', 'wiring', 'electrician'],
            'plumbing': ['plumbing', 'plumber', 'pipes', 'piping'],
            'framing': ['framing', 'frame', 'lumber'],
            'demolition': ['demolition', 'demo', 'tear down', 'removal'],
            'labor': ['labor', 'labour', 'work', 'crew', 'worker'],
            'material': ['material', 'materials', 'supplies', 'items'],
            'subcontractor': ['subcontractor', 'sub', 'contractor'],
            'painting': ['painting', 'paint', 'painter'],
            'flooring': ['flooring', 'floor', 'floors'],
            'roofing': ['roofing', 'roof', 'roofer'],
            'cabinet': ['cabinet', 'cabinets', 'cabinetry'],
            'tile': ['tile', 'tiles', 'tiling'],
            'siding': ['siding', 'exterior'],
            'trim': ['trim', 'molding', 'moulding'],
            'window': ['window', 'windows'],
            'door': ['door', 'doors'],
            'permit': ['permit', 'permits', 'fee', 'fees'],
            'foundation': ['foundation', 'footing', 'footings'],
            'hvac': ['hvac', 'heating', 'cooling', 'ventilation'],
            'insulation': ['insulation', 'insulate'],
            'drywall': ['drywall', 'sheetrock', 'gypsum'],
            'deck': ['deck', 'decking'],
            'gutter': ['gutter', 'gutters'],
            'staging': ['staging', 'scaffold', 'scaffolding']
        }
    
    def _preprocess_query(self, question: str) -> Dict[str, Any]:
        """Preprocess query to extract intent and expand keywords"""
        question_lower = question.lower()
        
        # Detect query intent
        intent = {
            'is_sum_query': any(word in question_lower for word in ['total', 'sum', 'amount', 'cost', 'how much']),
            'is_comparison': any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference between']),
            'is_list_query': any(word in question_lower for word in ['show', 'list', 'all', 'what are']),
            'is_detail_query': any(word in question_lower for word in ['details', 'detail', 'what is in', 'describe']),
            'is_search_query': True  # Default
        }
        
        # Extract keywords and expand them
        expanded_keywords = set()
        original_keywords = set()
        
        words = re.findall(r'\b\w+\b', question_lower)
        for word in words:
            if len(word) > 2:  # Skip very short words
                original_keywords.add(word)
                
                # Check if word matches any synonym group
                for key, synonyms in self.query_synonyms.items():
                    if word in synonyms or word == key:
                        expanded_keywords.update(synonyms)
                        break
        
        # Add original keywords to expanded set
        expanded_keywords.update(original_keywords)
        
        # Extract specific entities
        entities = {
            'row_number': self._extract_row_number(question),
            'area': self._extract_area_mention(question_lower),
            'cost_code_hint': self._extract_cost_code_hint(question_lower)
        }
        
        return {
            'original_question': question,
            'intent': intent,
            'expanded_keywords': list(expanded_keywords),
            'original_keywords': list(original_keywords),
            'entities': entities
        }
    
    def _extract_row_number(self, question: str) -> Optional[int]:
        """Extract row number from query"""
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
    
    def _extract_area_mention(self, question_lower: str) -> Optional[str]:
        """Extract area mentions from query"""
        area_keywords = ['house renovations', 'mudroom', 'roof deck', 'kitchen', 'bathroom', 'bedroom']
        for area in area_keywords:
            if area in question_lower:
                return area.title()
        return None
    
    def _extract_cost_code_hint(self, question_lower: str) -> Optional[str]:
        """Extract cost code hints from query"""
        # Map common terms to cost code patterns
        code_hints = {
            'cleanup': '203',
            'clean up': '203',
            'demolition': '205',
            'demo': '205',
            'electrical': '503',
            'plumbing': '508',
            'framing': '41',
            'painting': '738',
            'flooring': '726',
            'roofing': '420',
            'cabinet': '704',
            'tile': '768',
            'siding': '642'
        }
        
        for term, code in code_hints.items():
            if term in question_lower:
                return code
        return None
    
    def _fuzzy_match_score(self, s1: str, s2: str) -> float:
        """Calculate fuzzy match score between two strings"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def _keyword_search_metadata(self, keywords: List[str], all_metadatas: List[Dict], all_docs: List[str]) -> List[Tuple[int, float]]:
        """Search metadata fields for keyword matches and return (index, score) tuples"""
        matches = []
        
        for idx, (metadata, doc_text) in enumerate(zip(all_metadatas, all_docs)):
            score = 0.0
            matches_found = []
            
            # Search in key fields
            searchable_fields = [
                metadata.get('cost_code', ''),
                metadata.get('costCode', ''),
                metadata.get('description', ''),
                metadata.get('task_scope', ''),
                metadata.get('area', ''),
                doc_text
            ]
            
            searchable_text = ' '.join(str(field) for field in searchable_fields).lower()
            
            # Check each keyword
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Exact match in text
                if keyword_lower in searchable_text:
                    score += 2.0
                    matches_found.append(keyword)
                
                # Fuzzy match on cost codes (more lenient)
                cost_code = str(metadata.get('cost_code', '')).lower()
                if cost_code and self._fuzzy_match_score(keyword_lower, cost_code) > 0.6:
                    score += 1.5
                    matches_found.append(f"fuzzy:{keyword}")
                
                # Partial match (word boundary)
                if re.search(r'\b' + re.escape(keyword_lower), searchable_text):
                    score += 1.0
            
            # Bonus for matching multiple keywords
            if len(matches_found) > 1:
                score *= 1.2
            
            if score > 0:
                matches.append((idx, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    async def query(self, question: str, n_results: int = 10, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhanced hybrid query with:
        - Query preprocessing and expansion
        - Vector search (semantic)
        - Keyword search (metadata)
        - Fuzzy matching
        - Smart aggregation
        """
        try:
            print(f"🔍 Processing query: {question}")
            
            # Step 1: Preprocess query
            query_info = self._preprocess_query(question)
            print(f"📝 Query intent: {query_info['intent']}")
            print(f"🔑 Keywords: {query_info['original_keywords']}")
            print(f"🔄 Expanded: {query_info['expanded_keywords'][:5]}...")  # Show first 5
            
            # Step 2: Get all documents for keyword search fallback
            all_data = self.collection.get()
            all_metadatas = all_data.get('metadatas', [])
            all_documents = all_data.get('documents', [])
            all_ids = all_data.get('ids', [])
            
            if not all_metadatas:
                return {
                    "answer": "No data available in the system. Please generate embeddings first.",
                    "sources": [],
                    "chunks": [],
                    "data_types_found": [],
                    "row_numbers_found": []
                }
            
            # Step 3: Multi-strategy retrieval
            
            # Strategy 1: Vector search with expanded query
            expanded_query = question
            if query_info['expanded_keywords']:
                # Add expanded keywords to help vector search
                expanded_query = question + " " + " ".join(query_info['expanded_keywords'][:5])
            
            # Build filter
            query_filter = self._build_query_filter(
                data_types=data_types,
                row_number=query_info['entities']['row_number'],
                is_allowance=self._detect_allowance_query(question),
                has_materials='material' in question.lower()
            )
            
            # Try vector search with lower threshold (get more results)
            vector_results_count = min(n_results * 2, 50)  # Get more candidates
            
            try:
                vector_results = self.collection.query(
                    query_texts=[expanded_query],
                    n_results=vector_results_count,
                    where=query_filter
                )
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
                vector_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
            
            # Strategy 2: Keyword search on metadata
            print(f"🔎 Performing keyword search on {len(all_metadatas)} documents...")
            keyword_matches = self._keyword_search_metadata(
                query_info['expanded_keywords'],
                all_metadatas,
                all_documents
            )
            
            print(f"📊 Vector results: {len(vector_results['documents'][0])}, Keyword matches: {len(keyword_matches)}")
            
            # Step 4: Combine and deduplicate results
            combined_results = self._combine_search_results(
                vector_results,
                keyword_matches,
                all_ids,
                all_documents,
                all_metadatas,
                n_results=n_results
            )
            
            if not combined_results['documents']:
                return {
                    "answer": f"I searched for information about '{question}' but couldn't find any matching data. The keywords I looked for were: {', '.join(query_info['original_keywords'])}. Please try rephrasing your question or check if the data has been embedded.",
                    "sources": [],
                    "chunks": [],
                    "data_types_found": [],
                    "row_numbers_found": []
                }
            
            # Step 5: Analyze results
            relevant_chunks = combined_results['documents']
            metadatas = combined_results['metadatas']
            
            print(f"✅ Found {len(relevant_chunks)} relevant results")
            
            data_types_found = set()
            document_types_found = set()
            row_numbers_found = set()
            granularities_found = set()
            
            # Create sources list
            sources = []
            for metadata in metadatas:
                data_type = metadata.get('data_type', 'unknown')
                data_types_found.add(data_type)
                document_types_found.add(metadata.get('document_type', 'unknown'))
                granularities_found.add(metadata.get('granularity', 'unknown'))
                
                if 'row_number' in metadata:
                    row_numbers_found.add(metadata['row_number'])
                
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
            
            # Step 6: Generate intelligent answer
            answer = await self._generate_intelligent_answer(
                question=question,
                query_info=query_info,
                relevant_chunks=relevant_chunks,
                metadatas=metadatas
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
            print(f"❌ Error in query: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _combine_search_results(
        self,
        vector_results: Dict,
        keyword_matches: List[Tuple[int, float]],
        all_ids: List[str],
        all_documents: List[str],
        all_metadatas: List[Dict],
        n_results: int = 10
    ) -> Dict[str, List]:
        """Combine vector and keyword search results, deduplicate, and rank"""
        
        # Track results by ID to avoid duplicates
        results_map = {}
        
        # Add vector results
        if vector_results['documents'] and vector_results['documents'][0]:
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0],
                vector_results['ids'][0]
            )):
                # Convert distance to similarity score (lower distance = higher similarity)
                vector_score = 1.0 / (1.0 + distance)
                
                results_map[doc_id] = {
                    'document': doc,
                    'metadata': metadata,
                    'vector_score': vector_score,
                    'keyword_score': 0.0,
                    'combined_score': vector_score * 0.7  # 70% weight to vector
                }
        
        # Add keyword results
        for idx, keyword_score in keyword_matches[:50]:  # Top 50 keyword matches
            doc_id = all_ids[idx]
            normalized_keyword_score = min(keyword_score / 10.0, 1.0)  # Normalize to 0-1
            
            if doc_id in results_map:
                # Update existing result
                results_map[doc_id]['keyword_score'] = normalized_keyword_score
                results_map[doc_id]['combined_score'] += normalized_keyword_score * 0.3  # 30% weight to keywords
            else:
                # Add new result from keyword search
                results_map[doc_id] = {
                    'document': all_documents[idx],
                    'metadata': all_metadatas[idx],
                    'vector_score': 0.0,
                    'keyword_score': normalized_keyword_score,
                    'combined_score': normalized_keyword_score * 0.3
                }
        
        # Sort by combined score
        sorted_results = sorted(
            results_map.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:n_results]
        
        # Format output
        return {
            'documents': [r[1]['document'] for r in sorted_results],
            'metadatas': [r[1]['metadata'] for r in sorted_results],
            'ids': [r[0] for r in sorted_results]
        }
    
    async def _generate_intelligent_answer(
        self,
        question: str,
        query_info: Dict[str, Any],
        relevant_chunks: List[str],
        metadatas: List[Dict]
    ) -> str:
        """Generate intelligent answer based on query intent"""
        
        try:
            intent = query_info['intent']
            
            # Build context with smart formatting
            context_parts = []
            
            # If it's a sum/amount query, prepare data for calculation
            if intent['is_sum_query']:
                context_parts.append("=== COST DATA FOR CALCULATION ===")
                total_amount = 0.0
                items_for_calculation = []
                
                for metadata, chunk in zip(metadatas, relevant_chunks):
                    # Extract amounts from metadata
                    amount_fields = ['total', 'total_cost', 'sale_price', 'budgeted_total']
                    for field in amount_fields:
                        if field in metadata and metadata[field]:
                            try:
                                amount = float(metadata[field])
                                total_amount += amount
                                
                                cost_code = metadata.get('cost_code', metadata.get('costCode', 'Unknown'))
                                description = metadata.get('description', '')[:80]
                                row_num = metadata.get('row_number', '')
                                
                                item_info = f"Row {row_num}: {cost_code} - ${amount:,.2f}"
                                if description:
                                    item_info += f" ({description})"
                                
                                items_for_calculation.append(item_info)
                                break
                            except (ValueError, TypeError):
                                continue
                
                context_parts.append(f"TOTAL AMOUNT: ${total_amount:,.2f}")
                context_parts.append(f"NUMBER OF ITEMS: {len(items_for_calculation)}")
                context_parts.append("\nITEM BREAKDOWN:")
                context_parts.extend(items_for_calculation[:15])  # Show top 15
                
                if len(items_for_calculation) > 15:
                    context_parts.append(f"... and {len(items_for_calculation) - 15} more items")
            
            # Add detailed context for all queries
            context_parts.append("\n=== DETAILED INFORMATION ===")
            for idx, (chunk, metadata) in enumerate(zip(relevant_chunks[:10], metadatas[:10]), 1):
                data_type = metadata.get('data_type', 'unknown').upper()
                granularity = metadata.get('granularity', 'unknown')
                job_name = metadata.get('job_name', 'Unknown')
                
                if granularity == 'row':
                    row_num = metadata.get('row_number', '?')
                    label = f"[{data_type} ROW {row_num} - {job_name}]"
                else:
                    label = f"[{data_type} SUMMARY - {job_name}]"
                
                context_parts.append(f"\n{label}")
                context_parts.append(chunk[:800])  # Limit chunk size
            
            context = "\n".join(context_parts)
            
            # Create specialized prompt based on intent
            if intent['is_sum_query']:
                instruction = """You are answering a question about TOTAL AMOUNTS/COSTS.

IMPORTANT:
1. Use the TOTAL AMOUNT shown in the "COST DATA FOR CALCULATION" section
2. List the individual items that make up this total
3. Specify the exact dollar amount with proper formatting (e.g., $1,234.56)
4. If multiple items match, sum them and show the breakdown
5. Include row numbers when available"""
            
            elif intent['is_comparison']:
                instruction = """You are answering a COMPARISON question.

IMPORTANT:
1. Compare the specific items mentioned in the question
2. Show amounts side-by-side
3. Calculate differences and percentages if relevant
4. Highlight which is higher/lower"""
            
            elif intent['is_list_query']:
                instruction = """You are answering a LIST/SHOW ALL question.

IMPORTANT:
1. List ALL relevant items found in the data
2. Include row numbers, descriptions, and amounts
3. Organize by category or area if applicable
4. Use bullet points or numbered lists"""
            
            else:
                instruction = """You are answering a general construction project question.

IMPORTANT:
1. Provide specific details from the data
2. Include row numbers, amounts, descriptions
3. Be precise and factual"""
            
            prompt = f"""{instruction}

Context from construction project data:
{context}

Question: {question}

Answer the question clearly and specifically based on the data provided above. If amounts are shown, include them in your answer with proper formatting.

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a construction project assistant. Answer questions accurately using the provided project data. Always include specific dollar amounts, row numbers, and descriptions when available."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"I found {len(relevant_chunks)} relevant items but encountered an error generating the detailed response: {str(e)}"
    
    # Keep all existing helper methods
    def _build_query_filter(
        self, 
        data_types: Optional[List[str]] = None,
        row_number: Optional[int] = None,
        is_allowance: bool = False,
        has_materials: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB query filter based on query intent"""
        filters = []
        
        if data_types:
            filters.append({"data_type": {"$in": data_types}})
        
        if row_number is not None:
            filters.append({"row_number": row_number})
        
        if is_allowance:
            filters.append({"is_allowance": True})
        
        if has_materials:
            filters.append({"has_materials": True})
        
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def _detect_allowance_query(self, question: str) -> bool:
        """Detect if query is asking about allowances"""
        allowance_keywords = ['allowance', 'allowances', 'contingency', 'contingencies']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in allowance_keywords)
    
    def _get_display_identifier(self, metadata: Dict[str, Any]) -> str:
        """Get appropriate display identifier based on document type"""
        granularity = metadata.get('granularity', 'unknown')
        
        if granularity == 'row':
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
            categories = metadata.get('categories', '') or metadata.get('areas', '')
            return categories if categories else 'Summary'
    
    def _format_cost_info(self, metadata: Dict[str, Any], data_type: str) -> Optional[str]:
        """Format cost information based on data type and granularity"""
        granularity = metadata.get('granularity', 'job')
        
        if data_type == 'estimate' and granularity == 'row':
            total = metadata.get('total', 0)
            budgeted = metadata.get('budgeted_total', 0)
            variance = metadata.get('variance', 0)
            return f"${total:,.2f} est. / ${budgeted:,.2f} budgeted (Δ ${variance:,.2f})"
        
        elif data_type == 'estimate' and granularity == 'job':
            estimated = metadata.get('total_estimated_cost', 0)
            budgeted = metadata.get('total_budgeted_cost', 0)
            return f"${estimated:,.2f} est. / ${budgeted:,.2f} budgeted"
        
        elif data_type == 'flooring_estimate' and granularity == 'row':
            cost = metadata.get('total_cost', 0)
            sale = metadata.get('sale_price', 0)
            profit = metadata.get('profit', 0)
            return f"Cost: ${cost:,.2f} | Sale: ${sale:,.2f} | Profit: ${profit:,.2f}"
        
        elif data_type == 'flooring_estimate' and granularity == 'job':
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
    
    # Keep all existing methods from original file
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process all Firebase job data with multi-granularity embeddings"""
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
            job_datasets = await fetch_all_job_complete_data(company_id)
            
            if not job_datasets:
                print("No job data found in Firebase")
                return stats
            
            print(f"📊 Found {len(job_datasets)} datasets to process")
            
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
                    
                    jobs_processed.add(f"{company_id_current}_{job_id}")
                    
                    if company_id_current not in stats['companies_processed']:
                        stats['companies_processed'].append(company_id_current)
                    
                    if data_type == 'estimate':
                        await self._process_estimate_multi_granularity(
                            job_dataset, documents_to_add, embeddings_to_add,
                            ids_to_add, metadatas_to_add, stats
                        )
                    
                    elif data_type == 'flooring_estimate':
                        await self._process_flooring_estimate_multi_granularity(
                            job_dataset, documents_to_add, embeddings_to_add,
                            ids_to_add, metadatas_to_add, stats
                        )
                    
                    elif data_type in ['consumed', 'schedule']:
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
            
            stats['total_jobs_processed'] = len(jobs_processed)
            
            if documents_to_add:
                print(f"📝 Adding {len(documents_to_add)} documents to ChromaDB...")
                
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
            
            end_time = datetime.now()
            stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            print(f"✅ Processing complete! Processed {stats['total_jobs_processed']} jobs")
            
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
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Manually add a document to the collection"""
        try:
            embedding = self.embedding_service.generate_embedding(text)
            doc_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            metadata['added_at'] = datetime.now().isoformat()
            metadata['document_type'] = metadata.get('document_type', 'manual')
            
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
        """Get list of available jobs in the system"""
        try:
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
            
            result = []
            for job_info in jobs_info.values():
                job_info['data_types'] = list(job_info['data_types'])
                result.append(job_info)
            
            return result
            
        except Exception as e:
            print(f"Error getting available jobs: {e}")
            return []
    
    async def get_data_types_summary(self) -> Dict[str, Any]:
        """Get summary of available data types"""
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
        """Get list of available cost codes"""
        try:
            all_data = self.collection.get()
            cost_codes = set()
            
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    code = metadata.get('cost_code', '') or metadata.get('costCode', '')
                    if code and code.strip():
                        cost_codes.add(code.strip())
                    
                    codes = metadata.get('cost_codes', '')
                    if codes:
                        for code in codes.split(', '):
                            if code and code.strip():
                                cost_codes.add(code.strip())
                    
                    area = metadata.get('area', '')
                    if area and area.strip():
                        cost_codes.add(f"Area: {area.strip()}")
            
            return sorted(list(cost_codes))
            
        except Exception as e:
            print(f"Error getting available cost codes: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            sample_data = self.collection.peek(limit=100)
            sample_metadata = sample_data.get('metadatas', []) if sample_data else []
            
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
                        
                        if data_type in ['estimate', 'flooring_estimate']:
                            if granularity == 'row':
                                stats_by_type[data_type]['row_count'] += 1
                            elif granularity == 'job':
                                stats_by_type[data_type]['summary_count'] += 1
                        
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
                    else:
                        stats_by_type['unknown']['count'] += 1
                        if job_id:
                            stats_by_type['unknown']['jobs'].add(job_id)
            
            for type_stats in stats_by_type.values():
                type_stats['unique_jobs'] = len(type_stats['jobs'])
                del type_stats['jobs']
            
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
            all_data = self.collection.get()
            
            if not all_data['ids']:
                return 0
            
            return len(all_data['ids'])
            
        except Exception as e:
            print(f"Error clearing old documents: {e}")
            return 0