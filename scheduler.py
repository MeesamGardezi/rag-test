import os
import asyncio
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from app.database import get_chroma_collection
from app.embedding_service import EmbeddingService
from app.rag_service import RAGService

load_dotenv()

# Global scheduler instance
scheduler = None
embedding_service = None
rag_service = None

def initialize_services():
    """Initialize services needed for scheduled tasks"""
    global embedding_service, rag_service
    
    if embedding_service is None:
        embedding_service = EmbeddingService()
    
    if rag_service is None:
        rag_service = RAGService(embedding_service)
    
    return embedding_service, rag_service

async def nightly_embedding_job():
    """Main nightly job to process Firebase data and create embeddings"""
    print(f"🌙 Starting nightly embedding job at {datetime.now()}")
    
    try:
        # Initialize services
        emb_service, rag_svc = initialize_services()
        
        # Clear old embeddings first (optional - remove if you want to keep history)
        # This prevents the collection from growing indefinitely
        clear_old_embeddings = os.getenv("CLEAR_OLD_EMBEDDINGS", "false").lower() == "true"
        if clear_old_embeddings:
            print("🗑️ Clearing old embeddings...")
            collection = get_chroma_collection()
            all_data = collection.get()
            if all_data['ids']:
                collection.delete(ids=all_data['ids'])
                print(f"Cleared {len(all_data['ids'])} old documents")
        
        # Process Firebase data
        company_id = os.getenv("DEFAULT_COMPANY_ID")  # Process specific company or None for all
        stats = await rag_svc.process_firebase_data(company_id)
        
        print("📊 Nightly job statistics:")
        print(f"  - Jobs processed: {stats['total_jobs_processed']}")
        print(f"  - Entries embedded: {stats['total_entries_embedded']}")
        print(f"  - Companies processed: {len(stats['companies_processed'])}")
        print(f"  - Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"  - Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("❌ Errors encountered:")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        print("✅ Nightly embedding job completed successfully")
        
        return stats
        
    except Exception as e:
        error_msg = f"❌ Nightly job failed: {str(e)}"
        print(error_msg)
        raise

def run_nightly_job():
    """Wrapper to run async nightly job in sync context"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async job
        loop.run_until_complete(nightly_embedding_job())
        
    except Exception as e:
        print(f"Error in scheduled job: {e}")
    finally:
        loop.close()

def start_scheduler():
    """Start the background scheduler"""
    global scheduler
    
    # Check if scheduler is enabled
    enable_scheduler = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"
    if not enable_scheduler:
        print("📅 Scheduler disabled by configuration")
        return
    
    if scheduler is not None and scheduler.running:
        print("📅 Scheduler already running")
        return
    
    try:
        # Create scheduler
        scheduler = BackgroundScheduler(timezone='UTC')
        
        # Get schedule from environment variable (default: 2 AM daily)
        schedule_cron = os.getenv("EMBEDDING_SCHEDULE", "0 2 * * *")  # "minute hour day month day_of_week"
        
        print(f"📅 Setting up nightly job with schedule: {schedule_cron}")
        
        # Parse cron expression
        cron_parts = schedule_cron.split()
        if len(cron_parts) != 5:
            raise ValueError(f"Invalid cron expression: {schedule_cron}")
        
        minute, hour, day, month, day_of_week = cron_parts
        
        # Add job to scheduler
        scheduler.add_job(
            func=run_nightly_job,
            trigger=CronTrigger(
                minute=minute,
                hour=hour, 
                day=day,
                month=month,
                day_of_week=day_of_week
            ),
            id='nightly_embedding_job',
            name='Process Firebase Data and Generate Embeddings',
            replace_existing=True,
            misfire_grace_time=30*60  # 30 minutes grace period
        )
        
        # Start scheduler
        scheduler.start()
        print(f"✅ Scheduler started! Next run: {scheduler.get_job('nightly_embedding_job').next_run_time}")
        
        # Print all scheduled jobs
        jobs = scheduler.get_jobs()
        print(f"📋 Scheduled jobs: {len(jobs)}")
        for job in jobs:
            print(f"  - {job.name}: {job.next_run_time}")
    
    except Exception as e:
        print(f"❌ Error starting scheduler: {e}")
        raise

def stop_scheduler():
    """Stop the background scheduler"""
    global scheduler
    
    if scheduler is not None and scheduler.running:
        scheduler.shutdown()
        print("⏹️ Scheduler stopped")
    else:
        print("📅 Scheduler not running")

def get_scheduler_status():
    """Get current scheduler status"""
    global scheduler
    
    if scheduler is None:
        return {"running": False, "jobs": []}
    
    jobs = []
    if scheduler.running:
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name, 
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
    
    return {
        "running": scheduler.running if scheduler else False,
        "jobs": jobs
    }

def trigger_manual_run():
    """Manually trigger the nightly job (for testing)"""
    print("🔄 Manually triggering nightly job...")
    
    try:
        # Run in background thread
        import threading
        
        def run_job():
            run_nightly_job()
        
        thread = threading.Thread(target=run_job)
        thread.daemon = True
        thread.start()
        
        return {"status": "started", "message": "Manual job triggered in background"}
        
    except Exception as e:
        error_msg = f"Error triggering manual job: {e}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

# Cleanup function for graceful shutdown
def cleanup_scheduler():
    """Cleanup scheduler on application shutdown"""
    stop_scheduler()
    print("🧹 Scheduler cleanup completed")