from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

from .client.rq_cilent import queue
from .queues.worker import process_query


app = FastAPI()

# 1. Define a schema for the incoming JSON body
class QueryRequest(BaseModel):
    query: str = Field(..., description="The chat query of the user")

@app.get('/')
def root():
    return {"status": 'Server is up and running'}

# 2. Accept the body using the schema
@app.post('/chat')
def chat(request_data: QueryRequest):
    # Extract the string from the Pydantic model
    user_query = request_data.query
    
    # Enqueue the background task
    job = queue.enqueue(process_query, user_query)
    
    return {"status": "queued", "job_id": job.id}

# 3. Fetching the result
@app.post('/job-status')
def get_result(
    # FIX: Use Body(...) instead of Field(...) for direct JSON input parameters
    job_id: str = Body(..., embed=True, description="The unique ID of the queued job")
):
    # Fetch the job from Redis
    job = queue.fetch_job(job_id=job_id)
    
    # Safety Check: If the job doesn't exist in Redis
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found or has expired from cache.")
    
    # Return context dynamically based on its lifecycle status
    if job.is_finished:
        return {
            "status": "completed", 
            "result": job.result  # job.result captures the return value of process_query
        }
    elif job.is_failed:
        return {
            "status": "failed", 
            "error": "The background worker failed to process this query."
        }
    else:
        return {
            "status": job.get_status(),  # Returns 'queued' or 'started'
            "result": None
        }