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
        elif data_type == 'flooring_estimate':
            return self._create_flooring_estimate_text_representation(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_text_representation(job_data)
        else:
            return f"Unknown data type: {data_type}"
    
    def create_estimate_row_text(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> str:
        """Create detailed text representation for a single estimate row with full context"""
        row_num = row_data.get('row_number', 'Unknown')
        area = row_data.get('area', 'General')
        task_scope = row_data.get('taskScope', 'Unknown')
        cost_code = row_data.get('costCode', 'Unknown')
        description = row_data.get('description', 'No description')
        units = row_data.get('units', '')
        qty = float(row_data.get('qty', 0))
        rate = float(row_data.get('rate', 0))
        total = float(row_data.get('total', 0))
        budgeted_rate = float(row_data.get('budgetedRate', 0))
        budgeted_total = float(row_data.get('budgetedTotal', 0))
        notes_remarks = row_data.get('notesRemarks', '')
        row_type = row_data.get('rowType', 'estimate')
        materials = row_data.get('materials', [])
        
        # Build comprehensive text representation
        text_parts = [
            f"ESTIMATE ROW #{row_num}",
            f"Job: {job_context.get('job_name', 'Unknown')}",
            f"Location: {job_context.get('site_location', 'Unknown')}",
            f"Client: {job_context.get('client_name', 'Unknown')}",
            f"",
            f"ROW DETAILS:",
            f"Area: {area}",
            f"Task Scope: {task_scope}",
            f"Cost Code: {cost_code}",
            f"Row Type: {row_type.upper()}",
            f"",
            f"DESCRIPTION:",
            f"{description}",
        ]
        
        # Add notes if present
        if notes_remarks and notes_remarks.strip():
            text_parts.extend([
                f"",
                f"NOTES/REMARKS:",
                f"{notes_remarks}"
            ])
        
        # Add quantity and pricing information
        text_parts.extend([
            f"",
            f"QUANTITY & PRICING:",
            f"Quantity: {qty:,.2f} {units}",
            f"Estimated Rate: ${rate:,.2f} per {units}" if units else f"Estimated Rate: ${rate:,.2f}",
            f"Estimated Total: ${total:,.2f}",
            f"Budgeted Rate: ${budgeted_rate:,.2f} per {units}" if units else f"Budgeted Rate: ${budgeted_rate:,.2f}",
            f"Budgeted Total: ${budgeted_total:,.2f}",
        ])
        
        # Calculate variance
        if budgeted_total > 0:
            variance = total - budgeted_total
            variance_pct = (variance / budgeted_total) * 100
            text_parts.append(f"Variance: ${variance:,.2f} ({variance_pct:+.1f}%)")
        
        # Add materials information if present
        if materials and len(materials) > 0:
            text_parts.extend([
                f"",
                f"MATERIALS ({len(materials)} items):"
            ])
            for idx, material in enumerate(materials, 1):
                if isinstance(material, dict):
                    mat_name = material.get('name', 'Unknown material')
                    mat_qty = material.get('quantity', 0)
                    mat_unit = material.get('unit', '')
                    mat_cost = material.get('cost', 0)
                    text_parts.append(f"  {idx}. {mat_name}: {mat_qty} {mat_unit} @ ${mat_cost}")
                else:
                    text_parts.append(f"  {idx}. {material}")
        
        return "\n".join(text_parts)
    
    def create_flooring_estimate_row_text(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> str:
        """Create detailed text representation for a single flooring estimate row"""
        row_num = row_data.get('row_number', 'Unknown')
        
        # Extract flooring-specific fields
        floor_type_id = row_data.get('floorTypeId', 'Unknown')
        vendor = row_data.get('vendor', 'Unknown')
        item_material_name = row_data.get('itemMaterialName', 'Unknown')
        brand = row_data.get('brand', '')
        unit = row_data.get('unit', '')
        measured_qty = float(row_data.get('measuredQty', 0))
        supplier_qty = float(row_data.get('supplierQty', 0))
        waste_factor = float(row_data.get('wasteFactor', 0))
        qty_including_waste = float(row_data.get('qtyIncludingWaste', 0))
        unit_price = float(row_data.get('unitPrice', 0))
        cost_price = float(row_data.get('costPrice', 0))
        tax_freight = float(row_data.get('taxFreight', 0))
        total_cost = float(row_data.get('totalCost', 0))
        sale_price = float(row_data.get('salePrice', 0))
        notes_remarks = row_data.get('notesRemarks', '')
        
        text_parts = [
            f"FLOORING ESTIMATE ROW #{row_num}",
            f"Job: {job_context.get('job_name', 'Unknown')}",
            f"Location: {job_context.get('site_location', 'Unknown')}",
            f"",
            f"FLOORING DETAILS:",
            f"Floor Type ID: {floor_type_id}",
            f"Item/Material: {item_material_name}",
            f"Brand: {brand}" if brand else "",
            f"Vendor: {vendor}",
            f"",
            f"QUANTITIES:",
            f"Measured Quantity: {measured_qty:,.2f} {unit}",
            f"Supplier Quantity: {supplier_qty:,.2f} {unit}",
            f"Waste Factor: {waste_factor:.1f}%",
            f"Qty Including Waste: {qty_including_waste:,.2f} {unit}",
            f"",
            f"PRICING:",
            f"Unit Price: ${unit_price:,.2f} per {unit}",
            f"Cost Price: ${cost_price:,.2f}",
            f"Tax/Freight: ${tax_freight:,.2f}",
            f"Total Cost: ${total_cost:,.2f}",
            f"Sale Price: ${sale_price:,.2f}",
        ]
        
        # Calculate profit margin
        if sale_price > 0 and total_cost > 0:
            profit = sale_price - total_cost
            margin_pct = (profit / sale_price) * 100
            text_parts.append(f"Profit Margin: ${profit:,.2f} ({margin_pct:.1f}%)")
        
        # Add notes if present
        if notes_remarks and notes_remarks.strip():
            text_parts.extend([
                f"",
                f"NOTES/REMARKS:",
                f"{notes_remarks}"
            ])
        
        return "\n".join([part for part in text_parts if part])  # Filter empty strings
    
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
        """Create comprehensive text representation for estimate data with row-level details"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            total_rows = job_data.get('total_rows', len(entries))
            
            text_parts = [
                f"ESTIMATE DATA SUMMARY",
                f"Job: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Project Description: {job_data.get('project_description', 'N/A')}",
                f"Estimate Type: {job_data.get('estimate_type', 'general')}",
                f"Total Rows: {total_rows}",
                f""
            ]
            
            # Group by area and calculate totals
            areas = {}
            total_estimated = 0.0
            total_budgeted = 0.0
            allowance_count = 0
            estimate_count = 0
            
            for entry in entries:
                area = entry.get('area', 'General')
                row_type = entry.get('rowType', 'estimate')
                total = float(entry.get('total', 0))
                budgeted_total = float(entry.get('budgetedTotal', 0))
                
                total_estimated += total
                total_budgeted += budgeted_total
                
                if row_type == 'allowance':
                    allowance_count += 1
                else:
                    estimate_count += 1
                
                if area not in areas:
                    areas[area] = {
                        'total_estimated': 0.0,
                        'total_budgeted': 0.0,
                        'row_count': 0,
                        'allowances': 0,
                        'rows': []
                    }
                
                areas[area]['total_estimated'] += total
                areas[area]['total_budgeted'] += budgeted_total
                areas[area]['row_count'] += 1
                if row_type == 'allowance':
                    areas[area]['allowances'] += 1
                
                # Store simplified row info
                areas[area]['rows'].append({
                    'row_num': entry.get('row_number', 0),
                    'task': entry.get('taskScope', 'Unknown'),
                    'cost_code': entry.get('costCode', 'Unknown'),
                    'description': entry.get('description', '')[:50],  # First 50 chars
                    'total': total,
                    'row_type': row_type
                })
            
            # Add overall summary
            text_parts.extend([
                f"OVERALL TOTALS:",
                f"Total Estimated: ${total_estimated:,.2f}",
                f"Total Budgeted: ${total_budgeted:,.2f}",
                f"Variance: ${total_budgeted - total_estimated:,.2f}",
                f"Estimate Rows: {estimate_count}",
                f"Allowance Rows: {allowance_count}",
                f""
            ])
            
            # Add area breakdowns with row details
            text_parts.append(f"BREAKDOWN BY AREA ({len(areas)} areas):")
            for area, area_data in sorted(areas.items()):
                text_parts.extend([
                    f"",
                    f"Area: {area}",
                    f"  Rows: {area_data['row_count']} (includes {area_data['allowances']} allowances)",
                    f"  Estimated: ${area_data['total_estimated']:,.2f}",
                    f"  Budgeted: ${area_data['total_budgeted']:,.2f}",
                    f"  Variance: ${area_data['total_budgeted'] - area_data['total_estimated']:,.2f}",
                    f"  Key Items:"
                ])
                
                # Show top 5 items by cost in this area
                sorted_rows = sorted(area_data['rows'], key=lambda x: x['total'], reverse=True)[:5]
                for row in sorted_rows:
                    row_type_label = " [ALLOWANCE]" if row['row_type'] == 'allowance' else ""
                    text_parts.append(
                        f"    Row {row['row_num']}: {row['task']} - {row['cost_code']} "
                        f"(${row['total']:,.2f}){row_type_label}"
                    )
                    if row['description']:
                        text_parts.append(f"      {row['description']}...")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating estimate text representation: {e}")
            return f"Estimate data processing error: {str(e)}"
    
    def _create_flooring_estimate_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create text representation for flooring estimate data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            total_rows = job_data.get('total_rows', len(entries))
            
            text_parts = [
                f"FLOORING ESTIMATE DATA",
                f"Job: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Total Flooring Rows: {total_rows}",
                f""
            ]
            
            # Calculate totals
            total_cost = 0.0
            total_sale = 0.0
            total_measured_qty = 0.0
            vendors = set()
            floor_types = set()
            
            for entry in entries:
                total_cost += float(entry.get('totalCost', 0))
                total_sale += float(entry.get('salePrice', 0))
                total_measured_qty += float(entry.get('measuredQty', 0))
                
                vendor = entry.get('vendor', 'Unknown')
                floor_type = entry.get('floorTypeId', 'Unknown')
                
                if vendor and vendor != 'Unknown':
                    vendors.add(vendor)
                if floor_type and floor_type != 'Unknown':
                    floor_types.add(floor_type)
            
            total_profit = total_sale - total_cost
            margin_pct = (total_profit / total_sale * 100) if total_sale > 0 else 0
            
            text_parts.extend([
                f"TOTALS:",
                f"Total Cost: ${total_cost:,.2f}",
                f"Total Sale Price: ${total_sale:,.2f}",
                f"Total Profit: ${total_profit:,.2f} ({margin_pct:.1f}% margin)",
                f"Total Measured Quantity: {total_measured_qty:,.2f}",
                f"",
                f"Vendors ({len(vendors)}): {', '.join(sorted(vendors))}",
                f"Floor Types ({len(floor_types)}): {', '.join(sorted(floor_types))}",
                f"",
                f"FLOORING ITEMS:"
            ])
            
            # List key items
            for entry in entries[:10]:  # Show first 10 items
                row_num = entry.get('row_number', 0)
                item = entry.get('itemMaterialName', 'Unknown')
                vendor = entry.get('vendor', 'Unknown')
                qty = float(entry.get('measuredQty', 0))
                unit = entry.get('unit', '')
                sale = float(entry.get('salePrice', 0))
                
                text_parts.append(f"Row {row_num}: {item} from {vendor} - {qty:,.2f} {unit} @ ${sale:,.2f}")
            
            if len(entries) > 10:
                text_parts.append(f"... and {len(entries) - 10} more items")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            print(f"Error creating flooring estimate text representation: {e}")
            return f"Flooring estimate data processing error: {str(e)}"
    
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
                    'row_num': entry.get('row_number', 0),
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
                    text_parts.append(f"  Row {task['row_num']}: {task['name']}: {task['start_date']} to {task['end_date']} ({task['progress']:.0f}% complete)")
            
            # Add significant subtasks (those with hours or progress)
            significant_subtasks = [t for t in subtasks if t['hours'] > 0 or t['progress'] > 0]
            if significant_subtasks:
                text_parts.append(f"\nKey Subtasks ({len(significant_subtasks)}):")
                for task in significant_subtasks[:10]:  # Limit to top 10
                    resource_str = f" | Resources: {', '.join(task['resources'])}" if task['resources'] else ""
                    text_parts.append(f"  Row {task['row_num']}: {task['name']}: {task['hours']:.1f}h planned, {task['consumed']:.1f}h consumed ({task['progress']:.0f}% complete){resource_str}")
            
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
        elif data_type == 'flooring_estimate':
            return self._create_flooring_estimate_metadata(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_metadata(job_data)
        else:
            return {'error': f'Unknown data type: {data_type}'}
    
    def create_estimate_row_metadata(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed metadata for a single estimate row"""
        row_num = row_data.get('row_number', 0)
        
        # Parse numerical values
        qty = float(row_data.get('qty', 0))
        rate = float(row_data.get('rate', 0))
        total = float(row_data.get('total', 0))
        budgeted_rate = float(row_data.get('budgetedRate', 0))
        budgeted_total = float(row_data.get('budgetedTotal', 0))
        
        # Calculate variance
        variance = total - budgeted_total
        variance_pct = (variance / budgeted_total * 100) if budgeted_total != 0 else 0
        
        # Get materials info
        materials = row_data.get('materials', [])
        has_materials = len(materials) > 0
        material_count = len(materials)
        
        # Handle lastUpdated datetime conversion
        last_updated = job_context.get('last_updated', '')
        if hasattr(last_updated, 'isoformat'):
            last_updated_str = last_updated.isoformat()
        elif last_updated:
            last_updated_str = str(last_updated)
        else:
            last_updated_str = ''
        
        return {
            # Identification
            'job_name': str(job_context.get('job_name', 'Unknown')),
            'company_id': str(job_context.get('company_id', '')),
            'job_id': str(job_context.get('job_id', '')),
            'row_number': int(row_num),
            'document_type': 'estimate_row',
            'data_type': 'estimate',
            'granularity': 'row',  # Indicates this is row-level data
            
            # Row details
            'area': str(row_data.get('area', '')),
            'task_scope': str(row_data.get('taskScope', '')),
            'cost_code': str(row_data.get('costCode', '')),
            'description': str(row_data.get('description', ''))[:200],  # Limit length
            'units': str(row_data.get('units', '')),
            'row_type': str(row_data.get('rowType', 'estimate')),
            'is_allowance': bool(row_data.get('rowType', '') == 'allowance'),
            
            # Numerical values
            'qty': float(qty),
            'rate': float(rate),
            'total': float(total),
            'budgeted_rate': float(budgeted_rate),
            'budgeted_total': float(budgeted_total),
            'variance': float(variance),
            'variance_pct': float(variance_pct),
            
            # Materials
            'has_materials': bool(has_materials),
            'material_count': int(material_count),
            
            # Context
            'client_name': str(job_context.get('client_name', '')),
            'site_location': str(job_context.get('site_location', '')),
            'last_updated': last_updated_str,
            'estimate_type': str(job_context.get('estimate_type', 'general'))
        }
    
    def create_flooring_estimate_row_metadata(self, row_data: Dict[str, Any], job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed metadata for a single flooring estimate row"""
        row_num = row_data.get('row_number', 0)
        
        # Parse numerical values
        measured_qty = float(row_data.get('measuredQty', 0))
        unit_price = float(row_data.get('unitPrice', 0))
        total_cost = float(row_data.get('totalCost', 0))
        sale_price = float(row_data.get('salePrice', 0))
        
        # Calculate profit
        profit = sale_price - total_cost
        margin_pct = (profit / sale_price * 100) if sale_price > 0 else 0
        
        return {
            # Identification
            'job_name': str(job_context.get('job_name', 'Unknown')),
            'company_id': str(job_context.get('company_id', '')),
            'job_id': str(job_context.get('job_id', '')),
            'row_number': int(row_num),
            'document_type': 'flooring_estimate_row',
            'data_type': 'flooring_estimate',
            'granularity': 'row',
            
            # Flooring details
            'floor_type_id': str(row_data.get('floorTypeId', '')),
            'vendor': str(row_data.get('vendor', '')),
            'item_material_name': str(row_data.get('itemMaterialName', '')),
            'brand': str(row_data.get('brand', '')),
            'unit': str(row_data.get('unit', '')),
            
            # Numerical values
            'measured_qty': float(measured_qty),
            'unit_price': float(unit_price),
            'total_cost': float(total_cost),
            'sale_price': float(sale_price),
            'profit': float(profit),
            'margin_pct': float(margin_pct),
            
            # Context
            'client_name': str(job_context.get('client_name', '')),
            'site_location': str(job_context.get('site_location', ''))
        }
    
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
            'data_type': 'consumed',
            'granularity': 'job'
        }
        
        # Add individual category totals as separate metadata fields
        for category, total in category_totals.items():
            field_name = f"category_{category.lower().replace('/', '_').replace(' ', '_')}_total"
            metadata[field_name] = float(total)
        
        return metadata
    
    def _create_estimate_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for estimate data (summary level)"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Calculate estimate totals
        total_estimated = 0.0
        total_budgeted = 0.0
        areas = set()
        task_scopes = set()
        cost_codes = set()
        estimate_count = 0
        allowance_count = 0
        
        for entry in entries:
            total_estimated += float(entry.get('total', 0))
            total_budgeted += float(entry.get('budgetedTotal', 0))
            areas.add(entry.get('area', 'General'))
            task_scopes.add(entry.get('taskScope', 'Unknown'))
            cost_codes.add(entry.get('costCode', 'Unknown'))
            
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
            'total_rows': int(job_data.get('total_rows', len(entries))),
            'total_estimated_cost': float(total_estimated),
            'total_budgeted_cost': float(total_budgeted),
            'budget_variance': float(total_budgeted - total_estimated),
            'areas_count': int(len(areas)),
            'areas': ", ".join(list(areas)[:10]),  # Limit to 10 areas
            'task_scopes': ", ".join(list(task_scopes)[:10]),
            'cost_codes': ", ".join(list(cost_codes)[:10]),
            'estimate_rows': int(estimate_count),
            'allowance_rows': int(allowance_count),
            'document_type': 'job_estimate_data',
            'data_type': 'estimate',
            'granularity': 'job',
            'estimate_type': str(job_data.get('estimate_type', 'general'))
        }
    
    def _create_flooring_estimate_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for flooring estimate data (summary level)"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Calculate totals
        total_cost = 0.0
        total_sale = 0.0
        total_measured_qty = 0.0
        vendors = set()
        floor_types = set()
        
        for entry in entries:
            total_cost += float(entry.get('totalCost', 0))
            total_sale += float(entry.get('salePrice', 0))
            total_measured_qty += float(entry.get('measuredQty', 0))
            
            vendor = entry.get('vendor', '')
            floor_type = entry.get('floorTypeId', '')
            
            if vendor:
                vendors.add(vendor)
            if floor_type:
                floor_types.add(floor_type)
        
        total_profit = total_sale - total_cost
        margin_pct = (total_profit / total_sale * 100) if total_sale > 0 else 0
        
        return {
            'job_name': str(job_name),
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'client_name': str(job_data.get('client_name', '')),
            'site_location': str(job_data.get('site_location', '')),
            'total_rows': int(job_data.get('total_rows', len(entries))),
            'total_cost': float(total_cost),
            'total_sale': float(total_sale),
            'total_profit': float(total_profit),
            'margin_pct': float(margin_pct),
            'total_measured_qty': float(total_measured_qty),
            'vendors_count': int(len(vendors)),
            'vendors': ", ".join(list(vendors)[:10]),
            'floor_types_count': int(len(floor_types)),
            'floor_types': ", ".join(list(floor_types)[:10]),
            'document_type': 'flooring_estimate_data',
            'data_type': 'flooring_estimate',
            'granularity': 'job'
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
            'total_rows': int(job_data.get('total_rows', len(entries))),
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
            'data_type': 'schedule',
            'granularity': 'job'
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
        

    