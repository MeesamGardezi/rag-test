import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import time
from datetime import datetime
import json

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
    def create_job_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Convert job data into text suitable for embedding based on data type"""
        data_type = job_data.get('data_type', 'consumed')
        
        if data_type == 'consumed':
            return self._create_consumed_text_representation(job_data)
        elif data_type == 'estimate':
            return self._create_estimate_text_representation(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_text_representation(job_data)
        else:
            return f"Unknown data type: {data_type}"
    
    def _create_consumed_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for consumed cost data"""
        try:
            entries = job_data.get('entries', [])
            job_name = "Unknown Job"
            
            # Try to extract job name from first entry
            if entries and len(entries) > 0:
                job_name = entries[0].get('job', 'Unknown Job')
            
            # Create comprehensive text representation
            text_parts = [
                f"CONSUMED COST DATA",
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
            text_parts.append(f"Total Consumed Cost: ${total_cost:,.2f}")
            
            # Add category summaries
            for category, items in categories.items():
                category_total = sum(item['amount'] for item in items)
                text_parts.append(f"\n{category} (${category_total:,.2f}):")
                
                for item in items:
                    if item['amount'] > 0:
                        text_parts.append(f"  - {item['cost_code']}: ${item['amount']:,.2f}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating consumed text representation: {e}")
            return f"Consumed data processing error: {str(e)}"
    
    def _create_estimate_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for estimate data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            text_parts = [
                f"ESTIMATE DATA",
                f"Job: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Project Description: {job_data.get('project_description', 'N/A')}",
                f"Total Estimate Rows: {len(entries)}"
            ]
            
            # Group estimates by area and task scope
            areas = {}
            total_estimated = 0.0
            total_budgeted = 0.0
            
            for entry in entries:
                area = entry.get('area', 'General')
                task_scope = entry.get('taskScope', 'Unknown')
                cost_code = entry.get('costCode', 'Unknown')
                description = entry.get('description', '')
                units = entry.get('units', '')
                qty = float(entry.get('qty', 0))
                rate = float(entry.get('rate', 0))
                total = float(entry.get('total', 0))
                budgeted_rate = float(entry.get('budgetedRate', 0))
                budgeted_total = float(entry.get('budgetedTotal', 0))
                row_type = entry.get('rowType', 'estimate')
                
                total_estimated += total
                total_budgeted += budgeted_total
                
                if area not in areas:
                    areas[area] = {
                        'total_estimated': 0.0,
                        'total_budgeted': 0.0,
                        'tasks': []
                    }
                
                areas[area]['total_estimated'] += total
                areas[area]['total_budgeted'] += budgeted_total
                areas[area]['tasks'].append({
                    'task_scope': task_scope,
                    'cost_code': cost_code,
                    'description': description,
                    'qty': qty,
                    'units': units,
                    'rate': rate,
                    'total': total,
                    'budgeted_total': budgeted_total,
                    'type': row_type
                })
            
            # Add summary totals
            text_parts.append(f"Total Estimated Cost: ${total_estimated:,.2f}")
            text_parts.append(f"Total Budgeted Cost: ${total_budgeted:,.2f}")
            variance = total_budgeted - total_estimated
            text_parts.append(f"Budget Variance: ${variance:,.2f}")
            
            # Add area breakdowns
            for area, area_data in areas.items():
                text_parts.append(f"\nArea: {area}")
                text_parts.append(f"  Estimated: ${area_data['total_estimated']:,.2f}")
                text_parts.append(f"  Budgeted: ${area_data['total_budgeted']:,.2f}")
                
                # Add significant tasks
                for task in area_data['tasks']:
                    if task['total'] > 0:
                        text_parts.append(f"  - {task['task_scope']}: {task['description']} | {task['qty']} {task['units']} @ ${task['rate']}/unit = ${task['total']:,.2f}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating estimate text representation: {e}")
            return f"Estimate data processing error: {str(e)}"
    
    def _create_schedule_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for schedule data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            text_parts = [
                f"SCHEDULE DATA",
                f"Job: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Schedule Last Updated: {job_data.get('schedule_last_updated', 'Unknown')}",
                f"Total Tasks: {len(entries)}"
            ]
            
            # Filter out empty tasks and categorize
            valid_tasks = [entry for entry in entries if entry.get('task', '').strip()]
            main_tasks = []
            subtasks = []
            
            project_start_date = None
            project_end_date = None
            total_hours = 0.0
            total_consumed = 0.0
            in_progress_tasks = 0
            completed_tasks = 0
            
            for entry in valid_tasks:
                task_name = entry.get('task', '').strip()
                if not task_name:
                    continue
                    
                is_main_task = entry.get('isMainTask', False)
                hours = float(entry.get('hours', 0))
                consumed = float(entry.get('consumed', 0))
                progress = float(entry.get('percentageComplete', 0))
                
                total_hours += hours
                total_consumed += consumed
                
                if progress > 0 and progress < 100:
                    in_progress_tasks += 1
                elif progress >= 100:
                    completed_tasks += 1
                
                # Track project date range
                start_date = self._parse_schedule_date(entry.get('startDate'))
                end_date = self._parse_schedule_date(entry.get('endDate'))
                
                if start_date:
                    if not project_start_date or start_date < project_start_date:
                        project_start_date = start_date
                
                if end_date:
                    if not project_end_date or end_date > project_end_date:
                        project_end_date = end_date
                
                task_info = {
                    'name': task_name,
                    'hours': hours,
                    'consumed': consumed,
                    'progress': progress,
                    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'TBD',
                    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'TBD',
                    'task_type': entry.get('taskType', 'labour'),
                    'resources': list(entry.get('resources', {}).keys())
                }
                
                if is_main_task:
                    main_tasks.append(task_info)
                else:
                    subtasks.append(task_info)
            
            # Add project summary
            text_parts.append(f"Project Duration: {project_start_date.strftime('%Y-%m-%d') if project_start_date else 'TBD'} to {project_end_date.strftime('%Y-%m-%d') if project_end_date else 'TBD'}")
            text_parts.append(f"Total Planned Hours: {total_hours:,.1f}")
            text_parts.append(f"Total Consumed Hours: {total_consumed:,.1f}")
            text_parts.append(f"Tasks in Progress: {in_progress_tasks}")
            text_parts.append(f"Completed Tasks: {completed_tasks}")
            
            # Add main tasks
            if main_tasks:
                text_parts.append(f"\nMain Tasks ({len(main_tasks)}):")
                for task in main_tasks:
                    text_parts.append(f"  - {task['name']}: {task['start_date']} to {task['end_date']} ({task['progress']:.0f}% complete)")
            
            # Add significant subtasks (those with hours or progress)
            significant_subtasks = [t for t in subtasks if t['hours'] > 0 or t['progress'] > 0]
            if significant_subtasks:
                text_parts.append(f"\nKey Subtasks ({len(significant_subtasks)}):")
                for task in significant_subtasks[:10]:  # Limit to top 10
                    resource_str = f" | Resources: {', '.join(task['resources'])}" if task['resources'] else ""
                    text_parts.append(f"  - {task['name']}: {task['hours']:.1f}h planned, {task['consumed']:.1f}h consumed ({task['progress']:.0f}% complete){resource_str}")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating schedule text representation: {e}")
            return f"Schedule data processing error: {str(e)}"
    
    def _parse_schedule_date(self, date_value) -> Optional[datetime]:
        """Parse various date formats from schedule data"""
        if not date_value:
            return None
            
        try:
            # Handle string dates (YYYY-MM-DD format)
            if isinstance(date_value, str):
                return datetime.strptime(date_value, '%Y-%m-%d')
            
            # Handle Firebase Timestamp objects
            if hasattr(date_value, 'to_dict'):
                # Firestore Timestamp
                return date_value.to_datetime()
            
            # Handle datetime objects
            if isinstance(date_value, datetime):
                return date_value
                
        except Exception as e:
            print(f"Error parsing date {date_value}: {e}")
        
        return None
    
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
        """Create metadata for the document based on data type"""
        data_type = job_data.get('data_type', 'consumed')
        
        if data_type == 'consumed':
            return self._create_consumed_metadata(job_data)
        elif data_type == 'estimate':
            return self._create_estimate_metadata(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_metadata(job_data)
        else:
            return {'error': f'Unknown data type: {data_type}'}
    
    def _create_consumed_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for consumed cost data"""
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
            last_updated_str = last_updated.isoformat()
        elif last_updated:
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
            'document_type': 'job_cost_data',
            'data_type': 'consumed'
        }
        
        # Add individual category totals as separate metadata fields
        for category, total in category_totals.items():
            field_name = f"category_{category.lower().replace('/', '_').replace(' ', '_')}_total"
            metadata[field_name] = float(total)
        
        return metadata
    
    def _create_estimate_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for estimate data"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Calculate estimate totals
        total_estimated = 0.0
        total_budgeted = 0.0
        areas = set()
        task_scopes = set()
        estimate_count = 0
        allowance_count = 0
        
        for entry in entries:
            total_estimated += float(entry.get('total', 0))
            total_budgeted += float(entry.get('budgetedTotal', 0))
            areas.add(entry.get('area', 'General'))
            task_scopes.add(entry.get('taskScope', 'Unknown'))
            
            row_type = entry.get('rowType', 'estimate')
            if row_type == 'estimate':
                estimate_count += 1
            elif row_type == 'allowance':
                allowance_count += 1
        
        return {
            'job_name': str(job_name),
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'client_name': str(job_data.get('client_name', '')),
            'site_location': str(job_data.get('site_location', '')),
            'total_entries': int(len(entries)),
            'total_estimated_cost': float(total_estimated),
            'total_budgeted_cost': float(total_budgeted),
            'budget_variance': float(total_budgeted - total_estimated),
            'areas_count': int(len(areas)),
            'areas': ", ".join(list(areas)[:10]),  # Limit to 10 areas
            'task_scopes': ", ".join(list(task_scopes)[:10]),
            'estimate_rows': int(estimate_count),
            'allowance_rows': int(allowance_count),
            'document_type': 'job_estimate_data',
            'data_type': 'estimate'
        }
    
    def _create_schedule_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for schedule data"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Filter and analyze tasks
        valid_tasks = [entry for entry in entries if entry.get('task', '').strip()]
        
        total_hours = 0.0
        total_consumed = 0.0
        main_task_count = 0
        subtask_count = 0
        completed_tasks = 0
        in_progress_tasks = 0
        task_types = set()
        resources = set()
        
        project_start_date = None
        project_end_date = None
        
        for entry in valid_tasks:
            hours = float(entry.get('hours', 0))
            consumed = float(entry.get('consumed', 0))
            progress = float(entry.get('percentageComplete', 0))
            
            total_hours += hours
            total_consumed += consumed
            
            if entry.get('isMainTask', False):
                main_task_count += 1
            else:
                subtask_count += 1
            
            if progress >= 100:
                completed_tasks += 1
            elif progress > 0:
                in_progress_tasks += 1
            
            task_types.add(entry.get('taskType', 'labour'))
            
            # Collect resources
            task_resources = entry.get('resources', {})
            if isinstance(task_resources, dict):
                resources.update(task_resources.keys())
            
            # Track project date range
            start_date = self._parse_schedule_date(entry.get('startDate'))
            end_date = self._parse_schedule_date(entry.get('endDate'))
            
            if start_date:
                if not project_start_date or start_date < project_start_date:
                    project_start_date = start_date
            
            if end_date:
                if not project_end_date or end_date > project_end_date:
                    project_end_date = end_date
        
        return {
            'job_name': str(job_name),
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'client_name': str(job_data.get('client_name', '')),
            'site_location': str(job_data.get('site_location', '')),
            'total_tasks': int(len(valid_tasks)),
            'main_tasks': int(main_task_count),
            'subtasks': int(subtask_count),
            'completed_tasks': int(completed_tasks),
            'in_progress_tasks': int(in_progress_tasks),
            'total_planned_hours': float(total_hours),
            'total_consumed_hours': float(total_consumed),
            'project_start_date': project_start_date.strftime('%Y-%m-%d') if project_start_date else '',
            'project_end_date': project_end_date.strftime('%Y-%m-%d') if project_end_date else '',
            'task_types': ", ".join(list(task_types)),
            'resources': ", ".join(list(resources)[:10]),  # Limit to 10 resources
            'document_type': 'job_schedule_data',
            'data_type': 'schedule'
        }
    
    def test_embedding(self) -> bool:
        """Test the embedding service with a simple text"""
        try:
            test_text = "Test construction job with electrical work costing $1000"
            embedding = self.generate_embedding(test_text)
            return len(embedding) > 0
        except Exception as e:
            print(f"Embedding test failed: {e}")
            return False