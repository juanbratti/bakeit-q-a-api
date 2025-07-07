# DOCS ================================================

"""
FastAPI application for the RAG system.
"""

# IMPORTS =============================================

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List
import logging

from rag import RAG
from orchestrator import AgentOrchestrator

# LOGGING =============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RESPONSE MODELS =====================================

class QueryResponse(BaseModel):
    query: str
    documents: List[str]
    response: str

class AgentResponse(BaseModel):
    response: str

# APP =================================================

app = FastAPI(
    title="BakeitQ&A",
    description="A dessert recipe assistant",
)

# Initialize RAG system
rag = RAG()

# Initialize Agent Orchestrator
agent_orchestrator = AgentOrchestrator()

# ENDPOINTS ============================================

@app.get("/")
async def root():
    return {
        "message": "This is BakeitQ&A",
        "description": "This is a dessert recipe assistant!",
        "endpoints": {
            "/query": "Ask questions using RAG",
            "/agent_query": "Ask questions using the Agent Orchestrator",
        }
    }

@app.get("/query", response_model=QueryResponse)
async def query_recipes(q: str = Query(..., description="Search query for recipes using RAG")):
    """Query the RAG system for recipes"""
    try:
        response = rag.query(q)
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent_query", response_model=AgentResponse)
async def agent_query_recipes(q: str = Query(..., description="Search query for recipes using the Agent Orchestrator")):
    """Query the Agent Orchestrator for recipes, with planning and reasoning"""
    try:
        response = await agent_orchestrator.run(q)
        return AgentResponse(response=response)
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))