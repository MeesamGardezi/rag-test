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

class QueryClassifier:
    """
    Micro LLM-based query classification using GPT-3.5-turbo
    Fast, cheap, and highly accurate query understanding
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"
        
        self.classification_prompt = """You are a construction project query classifier. Analyze the user's question and return ONLY a JSON object with this exact structure:

{
  "data_type": "consumed" | "estimate" | "schedule" | "flooring_estimate" | "mixed",
  "confidence": 0.0 to 1.0,
  "intent": "sum" | "list" | "comparison" | "detail" | "search",
  "entities": {
    "row_number": null or integer,
    "area": null or string,
    "cost_code": null or string,
    "job_name": null or string
  },
  "reasoning": "brief explanation"
}

DATA TYPE DEFINITIONS:
- "consumed": Actual costs SPENT/PAID/USED (keywords: consumed, spent, actual, paid, used, expended, to date, so far, already spent)
- "estimate": Projected/planned costs (keywords: estimate, estimated, projection, projected, budget, budgeted, planned, forecasted, proposal, quote)
- "schedule": Timeline/dates/tasks (keywords: schedule, timeline, when, deadline, duration, date, start, finish, complete)
- "flooring_estimate": Flooring-specific estimates (keywords: flooring, floor, carpet, tile flooring, floor covering)
- "mixed": Multiple data types explicitly requested (e.g., "compare consumed vs estimate")

INTENT TYPES:
- "sum": Calculating totals (keywords: total, sum, how much, amount, cost)
- "list": Showing all items (keywords: list, show, all, what are, show me)
- "comparison": Comparing items (keywords: vs, versus, compare, difference, compared to)
- "detail": Getting details (keywords: details, what is, describe, what's in, tell me about)
- "search": General search/question

CRITICAL RULES:
1. If user says "spent", "paid", "used", "consumed" = ALWAYS "consumed" data type
2. If user says "estimate", "estimated", "budget", "budgeted" = ALWAYS "estimate" data type
3. Past tense (e.g., "what did we spend") = "consumed"
4. Future/planning tense (e.g., "what will cost") = "estimate"
5. If asking for BOTH types explicitly = "mixed"
6. Row numbers: extract any number after "row", "line", "item", "entry"

EXAMPLES:

Question: "What was the total consumed cost for electrical work?"
Response:
{
  "data_type": "consumed",
  "confidence": 0.98,
  "intent": "sum",
  "entities": {"row_number": null, "area": null, "cost_code": "electrical", "job_name": null},
  "reasoning": "Strong 'consumed' keyword + 'total' indicates sum query on actual spent costs. 'electrical work' is the cost code."
}

Question: "Show me the estimated cost for plumbing"
Response:
{
  "data_type": "estimate",
  "confidence": 0.95,
  "intent": "search",
  "entities": {"row_number": null, "area": null, "cost_code": "plumbing", "job_name": null},
  "reasoning": "'estimated cost' clearly indicates projected costs, not actual. Plumbing is the cost code."
}

Question: "How much did we spend on cleanup?"
Response:
{
  "data_type": "consumed",
  "confidence": 0.95,
  "intent": "sum",
  "entities": {"row_number": null, "area": null, "cost_code": "cleanup", "job_name": null},
  "reasoning": "'spend' (past tense 'did') indicates actual consumed costs. Cleanup is the cost code."
}

Question: "What's in estimate row 5?"
Response:
{
  "data_type": "estimate",
  "confidence": 0.98,
  "intent": "detail",
  "entities": {"row_number": 5, "area": null, "cost_code": null, "job_name": null},
  "reasoning": "Explicitly asks for 'estimate row' details. Row number 5 extracted."
}

Question: "Compare consumed vs estimated costs for electrical"
Response:
{
  "data_type": "mixed",
  "confidence": 0.98,
  "intent": "comparison",
  "entities": {"row_number": null, "area": null, "cost_code": "electrical", "job_name": null},
  "reasoning": "Explicitly compares both consumed AND estimated data types. This requires mixed data."
}

Question: "What are all the subcontractor costs in the budget?"
Response:
{
  "data_type": "estimate",
  "confidence": 0.90,
  "intent": "list",
  "entities": {"row_number": null, "area": null, "cost_code": "subcontractor", "job_name": null},
  "reasoning": "'budget' indicates estimate data (planned costs), not consumed. 'what are all' is a list query."
}

Question: "Total spent on kitchen renovation"
Response:
{
  "data_type": "consumed",
  "confidence": 0.95,
  "intent": "sum",
  "entities": {"row_number": null, "area": "kitchen", "cost_code": "renovation", "job_name": null},
  "reasoning": "'spent' is past tense indicating consumed/actual costs. Kitchen is the area."
}

Question: "What will the framing cost?"
Response:
{
  "data_type": "estimate",
  "confidence": 0.90,
  "intent": "search",
  "entities": {"row_number": null, "area": null, "cost_code": "framing", "job_name": null},
  "reasoning": "'will cost' is future tense indicating estimated/planned costs, not yet spent."
}

Now analyze this question and return ONLY valid JSON:"""
    
    def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Use GPT-3.5-turbo to classify query with high accuracy
        Returns structured classification in ~300ms
        """
        try:
            print(f"\n🤖 MICRO LLM CLASSIFICATION (GPT-3.5-turbo):")
            print(f"   Question: {question}")
            
            start_time = datetime.now()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise query classifier. Return only valid JSON. Never add explanatory text outside the JSON."},
                    {"role": "user", "content": f"{self.classification_prompt}\n\nQuestion: \"{question}\""}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            
            classification = json.loads(response.choices[0].message.content)
            
            print(f"   ✅ Data Type: {classification['data_type']} (confidence: {classification['confidence']:.2f})")
            print(f"   ✅ Intent: {classification['intent']}")
            print(f"   ✅ Entities: {classification['entities']}")
            print(f"   ✅ Reasoning: {classification['reasoning']}")
            print(f"   ⚡ Classification time: {elapsed:.0f}ms")
            
            return classification
            
        except Exception as e:
            print(f"   ❌ Classification failed: {e}")
            # Fallback to safe defaults
            return {
                "data_type": "mixed",
                "confidence": 0.3,
                "intent": "search",
                "entities": {
                    "row_number": None,
                    "area": None,
                    "cost_code": None,
                    "job_name": None
                },
                "reasoning": "Fallback due to classification error"
            }

class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = get_chroma_collection()
        
        # Initialize micro LLM classifier
        self.query_classifier = QueryClassifier()
        
        # Keyword synonyms for query expansion (backup)
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
    
    def _extract_keywords_from_entities(self, entities: Dict[str, Any]) -> List[str]:
        """Extract searchable keywords from LLM-detected entities"""
        keywords = []
        
        if entities.get('cost_code'):
            code = entities['cost_code']
            keywords.append(code)
            # Add synonyms if available
            if code in self.query_synonyms:
                keywords.extend(self.query_synonyms[code][:3])
        
        if entities.get('area'):
            keywords.append(entities['area'])
        
        if entities.get('job_name'):
            keywords.append(entities['job_name'])
        
        return keywords
    
    def _filter_results_by_data_type(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
        required_data_type: str,
        allow_other_types: bool = False
    ) -> Dict[str, List]:
        """
        Post-retrieval filtering to ensure ONLY correct data type
        This is the safety net that prevents wrong data from reaching GPT-4
        """
        filtered_docs = []
        filtered_metas = []
        filtered_ids = []
        
        print(f"\n🔍 POST-RETRIEVAL FILTERING:")
        print(f"   Required data type: {required_data_type}")
        print(f"   Allow other types: {allow_other_types}")
        print(f"   Input documents: {len(documents)}")
        
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_data_type = meta.get('data_type', 'unknown')
            
            # Handle "mixed" - allow all types
            if required_data_type == 'mixed':
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
                print(f"   ✅ KEPT (mixed mode): {doc_data_type}")
            elif doc_data_type == required_data_type:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
                print(f"   ✅ KEPT (exact match): {doc_data_type}")
            elif allow_other_types:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(doc_id)
                print(f"   ⚠️  KEPT (other allowed): {doc_data_type}")
            else:
                print(f"   ❌ FILTERED OUT: {doc_data_type}")
        
        print(f"   Output documents: {len(filtered_docs)}\n")
        
        return {
            'documents': filtered_docs,
            'metadatas': filtered_metas,
            'ids': filtered_ids
        }
    
    def _prepare_calculation_data(
        self,
        metadatas: List[Dict],
        relevant_chunks: List[str],
        intent: str
    ) -> Dict[str, Any]:
        """
        Prepare precise calculation data INCLUDING ALL MATCHING ROWS
        CRITICAL: This must include EVERY row, even duplicates with same cost code
        """
        calculation_data = {
            'items': [],
            'total_estimated': 0.0,
            'total_budgeted': 0.0,
            'total_consumed': 0.0,
            'total_variance': 0.0,
            'row_level_items': [],
            'job_level_items': [],
            'data_sources': set()
        }
        
        # Track which jobs have row-level data to prevent double counting
        jobs_with_rows = set()
        
        for metadata in metadatas:
            granularity = metadata.get('granularity', 'unknown')
            data_type = metadata.get('data_type', 'unknown')
            job_id = metadata.get('job_id', '')
            
            if granularity == 'row' and data_type in ['estimate', 'flooring_estimate']:
                jobs_with_rows.add(job_id)
        
        print(f"\n💰 PREPARING CALCULATION DATA:")
        print(f"   Total metadata items to process: {len(metadatas)}")
        
        # Process EVERY item - DO NOT skip duplicates
        for idx, (metadata, chunk) in enumerate(zip(metadatas, relevant_chunks), 1):
            data_type = metadata.get('data_type', 'unknown')
            granularity = metadata.get('granularity', 'unknown')
            job_id = metadata.get('job_id', '')
            job_name = metadata.get('job_name', 'Unknown')
            
            calculation_data['data_sources'].add(data_type)
            
            # ESTIMATE ROW LEVEL DATA - Include ALL rows
            if data_type == 'estimate' and granularity == 'row':
                row_num = metadata.get('row_number', '?')
                area = metadata.get('area', '')
                task_scope = metadata.get('task_scope', '')
                cost_code = metadata.get('cost_code', metadata.get('costCode', ''))
                description = metadata.get('description', '')[:100]
                
                estimated = float(metadata.get('total', 0))
                budgeted = float(metadata.get('budgeted_total', 0))
                variance = float(metadata.get('variance', 0))
                
                qty = float(metadata.get('qty', 0))
                rate = float(metadata.get('rate', 0))
                units = metadata.get('units', '')
                is_allowance = metadata.get('is_allowance', False)
                
                item = {
                    'type': 'estimate_row',
                    'job_name': job_name,
                    'row_number': row_num,
                    'area': area,
                    'task_scope': task_scope,
                    'cost_code': cost_code,
                    'description': description,
                    'estimated': estimated,
                    'budgeted': budgeted,
                    'variance': variance,
                    'qty': qty,
                    'rate': rate,
                    'units': units,
                    'is_allowance': is_allowance
                }
                
                calculation_data['row_level_items'].append(item)
                calculation_data['total_estimated'] += estimated
                calculation_data['total_budgeted'] += budgeted
                calculation_data['total_variance'] += variance
                calculation_data['items'].append(item)
                
                print(f"   [{idx}] INCLUDED Row #{row_num}: {cost_code} = ${estimated:,.2f}")
            
            # FLOORING ESTIMATE ROW LEVEL DATA - Include ALL rows
            elif data_type == 'flooring_estimate' and granularity == 'row':
                row_num = metadata.get('row_number', '?')
                item_name = metadata.get('item_material_name', '')
                vendor = metadata.get('vendor', '')
                
                cost = float(metadata.get('total_cost', 0))
                sale = float(metadata.get('sale_price', 0))
                profit = float(metadata.get('profit', 0))
                
                qty = float(metadata.get('measured_qty', 0))
                unit = metadata.get('unit', '')
                
                item = {
                    'type': 'flooring_row',
                    'job_name': job_name,
                    'row_number': row_num,
                    'item_name': item_name,
                    'vendor': vendor,
                    'cost': cost,
                    'sale': sale,
                    'profit': profit,
                    'qty': qty,
                    'unit': unit
                }
                
                calculation_data['row_level_items'].append(item)
                calculation_data['total_estimated'] += sale
                calculation_data['items'].append(item)
                
                print(f"   [{idx}] INCLUDED Flooring Row #{row_num}: {item_name} = ${sale:,.2f}")
            
            # CONSUMED DATA (job-level only)
            elif data_type == 'consumed' and granularity == 'job':
                consumed = float(metadata.get('total_cost', 0))
                categories = metadata.get('categories', '')
                
                item = {
                    'type': 'consumed_summary',
                    'job_name': job_name,
                    'consumed': consumed,
                    'categories': categories
                }
                
                calculation_data['job_level_items'].append(item)
                calculation_data['total_consumed'] += consumed
                calculation_data['items'].append(item)
                
                print(f"   [{idx}] INCLUDED Consumed: {job_name} = ${consumed:,.2f}")
            
            # JOB-LEVEL ESTIMATES (only if no row data exists - prevents double counting)
            elif granularity == 'job' and job_id not in jobs_with_rows:
                if data_type == 'estimate':
                    estimated = float(metadata.get('total_estimated_cost', 0))
                    budgeted = float(metadata.get('total_budgeted_cost', 0))
                    variance = float(metadata.get('budget_variance', 0))
                    
                    item = {
                        'type': 'estimate_summary',
                        'job_name': job_name,
                        'estimated': estimated,
                        'budgeted': budgeted,
                        'variance': variance,
                        'total_rows': metadata.get('total_rows', 0)
                    }
                    
                    calculation_data['job_level_items'].append(item)
                    calculation_data['total_estimated'] += estimated
                    calculation_data['total_budgeted'] += budgeted
                    calculation_data['total_variance'] += variance
                    calculation_data['items'].append(item)
                    
                    print(f"   [{idx}] INCLUDED Estimate Summary: {job_name} = ${estimated:,.2f}")
        
        print(f"\n   ✅ TOTAL ITEMS INCLUDED: {len(calculation_data['items'])}")
        print(f"   ✅ Row-level items: {len(calculation_data['row_level_items'])}")
        print(f"   ✅ Job-level items: {len(calculation_data['job_level_items'])}")
        print(f"   💵 GRAND TOTAL ESTIMATED: ${calculation_data['total_estimated']:,.2f}")
        print(f"   💵 GRAND TOTAL BUDGETED: ${calculation_data['total_budgeted']:,.2f}")
        print(f"   💵 GRAND TOTAL CONSUMED: ${calculation_data['total_consumed']:,.2f}\n")
        
        calculation_data['data_sources'] = list(calculation_data['data_sources'])
        return calculation_data
    
    def _format_calculation_context(self, calc_data: Dict[str, Any]) -> str:
        """Format calculation data into crystal-clear context for GPT-4"""
        context_parts = []
        
        context_parts.append("=" * 80)
        context_parts.append("PRECISE CALCULATION DATA - ALL MATCHING ROWS INCLUDED")
        context_parts.append("=" * 80)
        
        context_parts.append("\n📊 GRAND TOTALS (sum of ALL items below):")
        
        if calc_data['total_estimated'] > 0 or calc_data['total_budgeted'] > 0:
            context_parts.append(f"  • Total Estimated:  ${calc_data['total_estimated']:,.2f}")
            context_parts.append(f"  • Total Budgeted:   ${calc_data['total_budgeted']:,.2f}")
            context_parts.append(f"  • Variance:         ${calc_data['total_variance']:,.2f}")
        
        if calc_data['total_consumed'] > 0:
            context_parts.append(f"  • Total Consumed (ACTUAL SPENT):   ${calc_data['total_consumed']:,.2f}")
        
        context_parts.append(f"\n  • Number of Items:  {len(calc_data['items'])}")
        context_parts.append(f"  • Data Sources:     {', '.join(calc_data['data_sources'])}")
        
        # ROW-LEVEL ESTIMATE ITEMS - Show ALL rows
        if calc_data['row_level_items']:
            estimate_rows = [item for item in calc_data['row_level_items'] if item['type'] == 'estimate_row']
            flooring_rows = [item for item in calc_data['row_level_items'] if item['type'] == 'flooring_row']
            
            if estimate_rows:
                context_parts.append(f"\n\n{'=' * 80}")
                context_parts.append(f"📋 ESTIMATE - ALL {len(estimate_rows)} MATCHING ROWS")
                context_parts.append("=" * 80)
                context_parts.append("IMPORTANT: Sum ALL rows below for the total")
                
                for item in estimate_rows:
                    context_parts.append(f"\n🔹 ROW #{item['row_number']}: {item['job_name']}")
                    context_parts.append(f"   Area: {item['area']}")
                    context_parts.append(f"   Task: {item['task_scope']}")
                    context_parts.append(f"   Cost Code: {item['cost_code']}")
                    if item['description']:
                        context_parts.append(f"   Description: {item['description']}")
                    
                    if item['is_allowance']:
                        context_parts.append(f"   ⚠️  TYPE: ALLOWANCE")
                    
                    context_parts.append(f"\n   💰 COSTS:")
                    if item['units']:
                        context_parts.append(f"      Quantity: {item['qty']:,.2f} {item['units']}")
                        context_parts.append(f"      Rate: ${item['rate']:,.2f} per {item['units']}")
                    
                    context_parts.append(f"      ➡️  Estimated Total:  ${item['estimated']:,.2f}")
                    context_parts.append(f"      Budgeted Total:   ${item['budgeted']:,.2f}")
                    context_parts.append(f"      Variance:         ${item['variance']:,.2f}")
            
            if flooring_rows:
                context_parts.append(f"\n\n{'=' * 80}")
                context_parts.append(f"🏠 FLOORING ESTIMATE - ALL {len(flooring_rows)} MATCHING ROWS")
                context_parts.append("=" * 80)
                context_parts.append("IMPORTANT: Sum ALL rows below for the total")
                
                for item in flooring_rows:
                    context_parts.append(f"\n🔹 ROW #{item['row_number']}: {item['job_name']}")
                    context_parts.append(f"   Item: {item['item_name']}")
                    context_parts.append(f"   Vendor: {item['vendor']}")
                    context_parts.append(f"   Quantity: {item['qty']:,.2f} {item['unit']}")
                    context_parts.append(f"\n   💰 PRICING:")
                    context_parts.append(f"      Cost Price:  ${item['cost']:,.2f}")
                    context_parts.append(f"      ➡️  Sale Price:  ${item['sale']:,.2f}")
                    context_parts.append(f"      Profit:      ${item['profit']:,.2f}")
        
        # JOB-LEVEL SUMMARIES
        if calc_data['job_level_items']:
            context_parts.append(f"\n\n{'=' * 80}")
            context_parts.append(f"📊 JOB-LEVEL SUMMARIES ({len(calc_data['job_level_items'])} jobs)")
            context_parts.append("=" * 80)
            
            for item in calc_data['job_level_items']:
                context_parts.append(f"\n🏗️  {item['job_name']}")
                
                if item['type'] == 'estimate_summary':
                    context_parts.append(f"   Type: ESTIMATE (Projected Costs)")
                    context_parts.append(f"   Total Rows: {item['total_rows']}")
                    context_parts.append(f"   Estimated:  ${item['estimated']:,.2f}")
                    context_parts.append(f"   Budgeted:   ${item['budgeted']:,.2f}")
                    context_parts.append(f"   Variance:   ${item['variance']:,.2f}")
                
                elif item['type'] == 'consumed_summary':
                    context_parts.append(f"   Type: CONSUMED (Actual Spent)")
                    context_parts.append(f"   ➡️  Total Consumed:   ${item['consumed']:,.2f}")
                    context_parts.append(f"   Categories: {item['categories']}")
        
        context_parts.append("\n" + "=" * 80)
        
        return "\n".join(context_parts)
    
    async def query(self, question: str, n_results: int = 10, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        SUPER SMART query processing with micro LLM classification
        RETRIEVES ALL MATCHING ROWS for accurate calculations
        """
        try:
            print(f"\n{'='*80}")
            print(f"🔍 PROCESSING QUERY: {question}")
            print(f"{'='*80}")
            
            # STEP 1: Use micro LLM to classify query (GPT-3.5-turbo)
            classification = self.query_classifier.classify_query(question)
            
            detected_data_type = classification['data_type']
            confidence = classification['confidence']
            intent = classification['intent']
            entities = classification['entities']
            
            # STEP 2: Determine filtering strategy based on confidence
            forced_data_types = None
            should_filter_strictly = False
            
            # For sum queries, we need to retrieve MORE results to get all matching rows
            if intent == 'sum':
                retrieval_multiplier = 5  # Get 5x more results for sum queries
                print(f"\n💡 SUM QUERY DETECTED - Retrieving {retrieval_multiplier}x more results to ensure ALL matching rows are included\n")
            else:
                retrieval_multiplier = 2
            
            if confidence >= 0.7 and detected_data_type != 'mixed':
                # High confidence - FORCE filter to detected type
                forced_data_types = [detected_data_type]
                should_filter_strictly = True
                print(f"\n🔒 HIGH CONFIDENCE ({confidence:.2f}) - FORCING FILTER: {forced_data_types}\n")
            elif data_types:
                # Use user-provided data types
                forced_data_types = data_types
                should_filter_strictly = True
                print(f"\n🔒 USING PROVIDED DATA TYPES: {forced_data_types}\n")
            else:
                print(f"\n🔓 LOW CONFIDENCE ({confidence:.2f}) - ALLOWING MIXED RESULTS\n")
            
            # STEP 3: Extract keywords from entities
            search_keywords = self._extract_keywords_from_entities(entities)
            print(f"🔑 Search keywords: {search_keywords}")
            
            # Extract row number filter if detected
            row_number_filter = entities.get('row_number')
            if row_number_filter:
                print(f"🔢 Row number filter: {row_number_filter}")
            
            # Get all documents
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
                    "row_numbers_found": [],
                    "classification": classification
                }
            
            # STEP 4: Build enhanced query with extracted keywords
            enhanced_query = question
            if search_keywords:
                enhanced_query = question + " " + " ".join(search_keywords[:5])
            
            # Build ChromaDB filter
            query_filter = self._build_query_filter(
                data_types=forced_data_types,
                row_number=row_number_filter
            )
            
            # STEP 5: Vector search with INCREASED results to get ALL matching rows
            vector_results_count = min(n_results * retrieval_multiplier * 3, 200)  # Get up to 200 results
            print(f"📊 Requesting {vector_results_count} results to ensure ALL matching rows are retrieved")
            
            try:
                vector_results = self.collection.query(
                    query_texts=[enhanced_query],
                    n_results=vector_results_count,
                    where=query_filter
                )
                print(f"📊 Vector search returned: {len(vector_results['documents'][0]) if vector_results['documents'] else 0} results")
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
                vector_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
            
            # STEP 6: Get results from vector search
            results = {
                'documents': vector_results['documents'][0] if vector_results['documents'] else [],
                'metadatas': vector_results['metadatas'][0] if vector_results['metadatas'] else [],
                'ids': vector_results['ids'][0] if vector_results['ids'] else []
            }
            
            # STEP 7: POST-FILTER - Critical safety layer
            if should_filter_strictly and detected_data_type != 'mixed':
                print(f"\n🚨 APPLYING STRICT POST-FILTER 🚨")
                results = self._filter_results_by_data_type(
                    results['documents'],
                    results['metadatas'],
                    results['ids'],
                    required_data_type=detected_data_type,
                    allow_other_types=False  # STRICT MODE
                )
            
            # CRITICAL: For sum queries, DON'T limit results - include ALL matching rows
            if intent == 'sum':
                relevant_chunks = results['documents']  # ALL results, no limiting
                metadatas = results['metadatas']  # ALL metadata, no limiting
                print(f"\n✅ SUM QUERY: Including ALL {len(relevant_chunks)} results for accurate calculation\n")
            else:
                # For non-sum queries, limit to requested number
                relevant_chunks = results['documents'][:n_results * retrieval_multiplier]
                metadatas = results['metadatas'][:n_results * retrieval_multiplier]
                print(f"\n✅ NON-SUM QUERY: Limited to {len(relevant_chunks)} results\n")
            
            if not relevant_chunks:
                return {
                    "answer": f"I detected you're asking about '{detected_data_type}' data with {confidence:.0%} confidence, but couldn't find any matching results. This data type may not be available in the system yet.",
                    "sources": [],
                    "chunks": [],
                    "data_types_found": [],
                    "row_numbers_found": [],
                    "classification": classification
                }
            
            print(f"✅ Final results to process: {len(relevant_chunks)}")
            
            # Analyze results
            data_types_found = set()
            document_types_found = set()
            row_numbers_found = set()
            granularities_found = set()
            
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
            
            print(f"📈 Data types in results: {data_types_found}")
            print(f"📈 Row-level documents: {len([m for m in metadatas if m.get('granularity') == 'row'])}")
            
            # STEP 8: Generate intelligent answer with classification awareness
            answer = await self._generate_intelligent_answer(
                question=question,
                classification=classification,
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
                "row_numbers_found": sorted(list(row_numbers_found)) if row_numbers_found else [],
                "classification": classification
            }
            
        except Exception as e:
            print(f"❌ Error in query: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _generate_intelligent_answer(
        self,
        question: str,
        classification: Dict[str, Any],
        relevant_chunks: List[str],
        metadatas: List[Dict]
    ) -> str:
        """Generate answer with classification-aware context"""
        
        try:
            intent = classification['intent']
            detected_type = classification['data_type']
            confidence = classification['confidence']
            
            # Data type context for GPT-4
            type_context = f"\n🎯 DETECTED: User is asking about {detected_type.upper()} data (confidence: {confidence:.0%})\n"
            
            # Build instruction based on intent
            if intent == 'sum':
                calc_data = self._prepare_calculation_data(metadatas, relevant_chunks, intent)
                context = self._format_calculation_context(calc_data)
                
                # Determine cost instruction based on data type
                if detected_type == 'consumed':
                    cost_instruction = "Use CONSUMED costs - these are ACTUAL SPENT amounts (the 'Total Consumed' values)"
                elif detected_type == 'estimate':
                    cost_instruction = "Use ESTIMATED costs - these are PROJECTED amounts (the 'Estimated Total' values)"
                elif detected_type == 'mixed':
                    cost_instruction = "Use the appropriate cost type based on what the user asked for"
                else:
                    cost_instruction = "Use the cost values shown in the data"
                
                instruction = f"""{type_context}
You are calculating a TOTAL/SUM for construction costs.

🚨 CRITICAL RULES 🚨:
1. The GRAND TOTAL is already calculated at the top: ${calc_data['total_estimated']:,.2f} (estimated) or ${calc_data['total_consumed']:,.2f} (consumed)
2. USE THIS GRAND TOTAL - it includes ALL {len(calc_data['items'])} matching items
3. {cost_instruction}
4. List ALL items that contribute to this total (show ALL row numbers)
5. Format ALL amounts as $1,234.56 (commas + 2 decimal places)
6. DO NOT recalculate - the sum is already correct in the data above

ANSWER FORMAT:
**Total: $X,XXX.XX** (use the GRAND TOTAL from the data above)

This total includes {len(calc_data['items'])} items:
- Row #X (identifier): $X,XXX.XX
- Row #Y (identifier): $X,XXX.XX
- Row #Z (identifier): $X,XXX.XX
[list ALL items]

IMPORTANT: List ALL {len(calc_data['items'])} items shown in the breakdown above."""

            elif intent == 'comparison':
                calc_data = self._prepare_calculation_data(metadatas, relevant_chunks, intent)
                context = self._format_calculation_context(calc_data)
                
                instruction = f"""{type_context}
You are comparing construction costs/data.

IMPORTANT:
1. Use the totals provided in the data above
2. Show side-by-side comparison
3. Calculate differences and percentages
4. Highlight which is higher/lower
5. Use exact values from data"""

            elif intent == 'list':
                context = "\n\n".join([
                    f"[{metadata.get('data_type', '').upper()}] {metadata.get('job_name', 'Unknown')}\n{chunk[:600]}"
                    for metadata, chunk in zip(metadatas[:50], relevant_chunks[:50])  # Show more for lists
                ])
                
                instruction = f"""{type_context}
You are listing construction items.

IMPORTANT:
1. List ALL {len(metadatas)} relevant items found
2. Include identifiers (row numbers, cost codes)
3. Include amounts where available
4. Organize clearly (use bullet points)"""
            
            elif intent == 'detail':
                context = "\n\n".join([
                    f"[{metadata.get('data_type', '').upper()}] {metadata.get('job_name', 'Unknown')}\n{chunk[:800]}"
                    for metadata, chunk in zip(metadatas[:10], relevant_chunks[:10])
                ])
                
                instruction = f"""{type_context}
You are providing detailed information about a specific item.

IMPORTANT:
1. Include all relevant details from the data
2. Show amounts, quantities, descriptions
3. Be comprehensive but concise"""
            
            else:  # search
                context = "\n\n".join([
                    f"[{metadata.get('data_type', '').upper()}] {metadata.get('job_name', 'Unknown')}\n{chunk[:800]}"
                    for metadata, chunk in zip(metadatas[:20], relevant_chunks[:20])
                ])
                
                instruction = f"""{type_context}
You are answering a construction project question.

IMPORTANT:
1. Provide specific, factual information from the data
2. Include relevant identifiers and amounts
3. Be precise and accurate"""
            
            prompt = f"""{instruction}

{context}

Question: {question}

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise construction project assistant. Use EXACT totals and values from the data provided. The calculations are already done correctly - use those numbers. Never recalculate or estimate. Always format dollar amounts with commas and two decimal places ($1,234.56). When showing breakdowns, list ALL items provided in the data."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,  # Increased for longer lists
                temperature=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return f"I found {len(relevant_chunks)} relevant items but encountered an error generating the answer: {str(e)}"
    
    # Helper methods
    def _build_query_filter(
        self, 
        data_types: Optional[List[str]] = None,
        row_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB query filter"""
        filters = []
        
        if data_types:
            filters.append({"data_type": {"$in": data_types}})
        
        if row_number is not None:
            filters.append({"row_number": row_number})
        
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def _get_display_identifier(self, metadata: Dict[str, Any]) -> str:
        """Get appropriate display identifier"""
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
        """Format cost information based on data type"""
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
    
    # Keep ALL existing data processing methods exactly as before
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process all Firebase job data with multi-granularity embeddings"""
        print("🔄 Starting comprehensive Firebase data processing...")
        
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
                    error_msg = f"Error processing {job_dataset.get('data_type', 'unknown')} for job {job_dataset.get('job_id', 'unknown')}: {str(e)}"
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
        """Process estimate data with job + row level embeddings"""
        job_id = job_dataset.get('job_id', 'unknown')
        company_id = job_dataset.get('company_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Job-level summary
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
        
        # Row-level details
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
        
        print(f"✅ Processed estimate for {job_id}: 1 summary + {len(entries)} rows")
    
    async def _process_flooring_estimate_multi_granularity(
        self, job_dataset: Dict[str, Any],
        documents: List[str], embeddings: List[List[float]],
        ids: List[str], metadatas: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        """Process flooring estimate with job + row level"""
        job_id = job_dataset.get('job_id', 'unknown')
        company_id = job_dataset.get('company_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Job-level
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
        
        # Row-level
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
        
        print(f"✅ Processed flooring for {job_id}: 1 summary + {len(entries)} rows")
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Manually add document"""
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
            
            print(f"✅ Manually added document: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"Error adding document: {e}")
            raise
    
    async def get_available_jobs(self) -> List[Dict[str, Any]]:
        """Get list of available jobs"""
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
            print(f"Error getting jobs: {e}")
            return []
    
    async def get_data_types_summary(self) -> Dict[str, Any]:
        """Get data types summary"""
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
            print(f"Error getting summary: {e}")
            return {'error': str(e)}
    
    async def get_available_cost_codes(self) -> List[str]:
        """Get available cost codes"""
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
            print(f"Error getting cost codes: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
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
                'schedule': {'count': 0, 'total_hours': 0.0, 'jobs': set()},
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
            print(f"Error getting stats: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def clear_old_documents(self, days_old: int = 7) -> int:
        """Clear old documents"""
        try:
            all_data = self.collection.get()
            
            if not all_data['ids']:
                return 0
            
            return len(all_data['ids'])
            
        except Exception as e:
            print(f"Error clearing: {e}")
            return 0