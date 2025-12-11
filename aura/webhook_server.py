"""
Webhook Server for Dialogflow CX
"""

import os
import json
import logging
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from google.genai import types
import uvicorn

# Load environment variables BEFORE importing ADK
from dotenv import load_dotenv
load_dotenv()

# Validate required env vars
required_vars = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# Set Vertex AI flag
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

# Now import ADK
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from app.agents.agent import root_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Global runner and session service
runner: Optional[Runner] = None
session_service: Optional[InMemorySessionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ADK runner on startup."""
    global runner, session_service
    
    logger.info("Initializing ADK Agent...")
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="policyvoice",
        session_service=session_service,
    )
    
    logger.info("ADK Agent ready to receive requests!")
    yield
    
    logger.info("Shutting down ADK Agent...")


app = FastAPI(
    title="ADK Webhook",
    description="Dialogflow CX webhook for ADK Agent",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "agent": "policyvoice"}


@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    """Handle Dialogflow CX webhook requests."""
    global runner, session_service
    
    try:
        body = await request.json()
        logger.info(f"Received webhook request: {json.dumps(body, indent=2)[:500]}...")
        
        # Extract session info
        session_info = body.get("sessionInfo", {})
        session_id = session_info.get("session", "unknown-session")
        parameters = session_info.get("parameters", {})
        
        # Extract user query
        user_query = (
            body.get("text") or 
            body.get("transcript") or
            body.get("fulfillmentInfo", {}).get("tag") or
            ""
        )
        
        if not user_query and body.get("intentInfo"):
            messages = body.get("messages", [])
            for msg in reversed(messages):
                if msg.get("source") == "VIRTUAL_AGENT":
                    continue
                user_query = msg.get("text", {}).get("text", [""])[0]
                if user_query:
                    break
        
        if not user_query:
            logger.warning("No user query found in request")
            return _build_response("I didn't catch that. Could you please repeat?", session_id, parameters)
        
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Extract customer identifiers
        customer_id = parameters.get("customer_id", "")
        policy_number = parameters.get("policy_number", "")
        caller_name = parameters.get("caller_name", "")
        
        # Clean session ID for ADK
        clean_session_id = session_id.split("/")[-1] if "/" in session_id else session_id
        user_id = f"dfcx-{clean_session_id[:20]}"
        
        # Get or create ADK session
        session = await session_service.get_session(
            app_name="policyvoice",
            user_id=user_id,
            session_id=clean_session_id
        )
        
        if session is None:
            session = await session_service.create_session(
                app_name="policyvoice",
                user_id=user_id,
                session_id=clean_session_id,
                state={
                    "customer_id": customer_id,
                    "policy_number": policy_number,
                    "caller_name": caller_name,
                    "session_id": clean_session_id,
                }
            )
            logger.info(f"Created new session: {clean_session_id}")
        else:
            logger.info(f"Retrieved existing session: {clean_session_id}")
        
        # Update session state with latest parameters
        session.state["customer_id"] = customer_id or session.state.get("customer_id", "")
        session.state["policy_number"] = policy_number or session.state.get("policy_number", "")
        session.state["caller_name"] = caller_name or session.state.get("caller_name", "")
        
        # Run the ADK agent
        response_text = ""
        should_escalate = False
        
        # Convert string to proper Content object
        user_content = types.Content(role='user', parts=[types.Part(text=user_query)])
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=clean_session_id,
            new_message=user_content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text
            
            if hasattr(event, 'actions') and event.actions:
                state_delta = getattr(event.actions, 'state_delta', {})
                if 'final_response' in state_delta:
                    try:
                        final = json.loads(state_delta['final_response'])
                        response_text = final.get('speech_text', response_text)
                        should_escalate = final.get('should_escalate', False)
                    except:
                        pass
        
        # Get final response from session state if not captured
        if not response_text:
            final_response = session.state.get("final_response", "")
            if final_response:
                try:
                    final = json.loads(final_response)
                    response_text = final.get('speech_text', '')
                    should_escalate = final.get('should_escalate', False)
                except:
                    response_text = final_response
        
        if not response_text:
            response_text = "I'm sorry, I couldn't process that request. Would you like to speak with an agent?"
        
        logger.info(f"Response: {response_text[:100]}...")
        
        return _build_response(
            text=response_text,
            session_id=session_id,
            parameters=parameters,
            should_escalate=should_escalate
        )
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return _build_response(
            "I'm experiencing a technical issue. Let me connect you with an agent who can help.",
            session_id="error",
            parameters={},
            should_escalate=True
        )


def _build_response(
    text: str,
    session_id: str,
    parameters: dict,
    should_escalate: bool = False
) -> JSONResponse:
    """Build Dialogflow CX webhook response."""
    response = {
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [text]}}]
        },
        "sessionInfo": {
            "session": session_id,
            "parameters": parameters
        }
    }
    
    if should_escalate:
        response["sessionInfo"]["parameters"]["escalate_to_agent"] = True
        response["targetPage"] = "projects/-/locations/-/agents/-/flows/-/pages/LIVE_AGENT_HANDOFF"
    
    return JSONResponse(content=response)


@app.post("/")
async def root_webhook(request: Request):
    """Alternative endpoint."""
    return await dialogflow_webhook(request)


# Simple test endpoint
@app.post("/test")
async def test_query(request: Request):
    """Test endpoint - send JSON with {"text": "your question"}"""
    body = await request.json()
    test_request = Request(scope=request.scope)
    test_request._json = {
        "text": body.get("text", "Hello"),
        "sessionInfo": {
            "session": "test-session-123",
            "parameters": body.get("parameters", {})
        }
    }
    # Manually set the json
    request._json = test_request._json
    return await dialogflow_webhook(request)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting webhook server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)