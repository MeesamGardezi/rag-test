import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
    def create_job_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Convert job data into text suitable for embedding"""
        try:
            entries = job_data.get('entries', [])
            job_name = "Unknown Job"
            
            # Try to extract job name from first entry
            if entries and len(entries) > 0:
                job_name = entries[0].get('job', 'Unknown Job')
            
            # Create comprehensive text representation
            text_parts = [
                f"Job: {job_name}",
                f"Last Updated: {job_data.get('lastUpdated', 'Unknown')}",
                f"Total Entries: {len(entries)}"
            ]
            
            # Group entries by category for better organization
            categories = {}
            total_cost = 0.0
            
            for entry in entries:
                cost_code = entry.get('costCode', 'Unknown')
                amount_str = entry.get('amount', '0')
                job_entry_name = entry.get('job', job_name)
                
                # Parse amount
                try:
                    amount = float(amount_str) if amount_str else 0.0
                    total_cost += amount
                except (ValueError, TypeError):
                    amount = 0.0
                
                # Categorize cost code
                category = self.categorize_cost_code(cost_code)
                
                if category not in categories:
                    categories[category] = []
                
                categories[category].append({
                    'cost_code': cost_code,
                    'amount': amount,
                    'description': cost_code.split(' ', 1)[-1] if ' ' in cost_code else cost_code
                })
            
            # Add total cost
            text_parts.append(f"Total Cost: ${total_cost:,.2f}")
            
            # Add category summaries
            for category, items in categories.items():
                category_total = sum(item['amount'] for item in items)
                text_parts.append(f"\n{category} (${category_total:,.2f}):")
                
                for item in items:
                    if item['amount'] > 0:
                        text_parts.append(f"  - {item['cost_code']}: ${item['amount']:,.2f}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating text representation: {e}")
            return f"Job data processing error: {str(e)}"
    
    def categorize_cost_code(self, cost_code: str) -> str:
        """Categorize cost codes with proper priority logic"""
        if not cost_code:
            return "Unknown"
            
        code_lower = cost_code.lower()
        
        # Priority 1: Check for "subcontractor" in the name first
        if 'subcontractor' in code_lower:
            return "Subcontractors"
        
        # Priority 2: Check suffix patterns in the code
        # Look for patterns like "503S", "110O", "414M", "108L"
        import re
        suffix_match = re.search(r'\d+([SMLO])\b', cost_code)
        if suffix_match:
            suffix = suffix_match.group(1).upper()
            if suffix == 'S':
                return "Subcontractors"
            elif suffix == 'M':
                return "Materials"
            elif suffix == 'L':
                return "Labor"
            elif suffix == 'O':
                return "Other/Overhead"
        
        # Priority 3: Check for keywords
        if any(word in code_lower for word in ['material', 'materials']):
            return "Materials"
        elif any(word in code_lower for word in ['labor', 'labour']):
            return "Labor"
        elif any(word in code_lower for word in ['permit', 'fee', 'overhead', 'management']):
            return "Other/Overhead"
        
        # Default fallback
        return "Other"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                # For failed batch, add empty embeddings to maintain alignment
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def create_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for the document"""
        entries = job_data.get('entries', [])
        
        # Extract job info
        job_name = entries[0].get('job', 'Unknown') if entries else 'Unknown'
        
        # Calculate totals by category
        category_totals = {}
        total_cost = 0.0
        cost_codes = set()
        
        for entry in entries:
            cost_code = entry.get('costCode', '')
            amount_str = entry.get('amount', '0')
            
            cost_codes.add(cost_code)
            
            try:
                amount = float(amount_str) if amount_str else 0.0
                total_cost += amount
                
                category = self.categorize_cost_code(cost_code)
                category_totals[category] = category_totals.get(category, 0.0) + amount
                
            except (ValueError, TypeError):
                continue
        
        # Convert lists to strings for ChromaDB compatibility
        categories_str = ", ".join(category_totals.keys()) if category_totals else ""
        cost_codes_str = ", ".join(list(cost_codes)[:10])  # Limit to first 10 codes
        
        # Handle lastUpdated datetime conversion
        last_updated = job_data.get('lastUpdated', '')
        if hasattr(last_updated, 'isoformat'):
            # Convert datetime object to ISO string
            last_updated_str = last_updated.isoformat()
        elif last_updated:
            # If it's already a string, use it as-is
            last_updated_str = str(last_updated)
        else:
            last_updated_str = ''
        
        # Convert category totals to individual metadata fields
        metadata = {
            'job_name': str(job_name) if job_name else 'Unknown',
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'last_updated': last_updated_str,
            'total_entries': int(len(entries)),
            'total_cost': float(total_cost),
            'categories': categories_str,
            'cost_codes': cost_codes_str,
            'document_type': 'job_cost_data'
        }
        
        # Add individual category totals as separate metadata fields
        for category, total in category_totals.items():
            # Replace spaces and special chars with underscores for field names
            field_name = f"category_{category.lower().replace('/', '_').replace(' ', '_')}_total"
            metadata[field_name] = float(total)
        
        return metadata
    
    def test_embedding(self) -> bool:
        """Test the embedding service with a simple text"""
        try:
            test_text = "Test construction job with electrical work costing $1000"
            embedding = self.generate_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            print(f"Embedding test failed: {e}")
            return False