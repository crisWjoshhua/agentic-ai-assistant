from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import time
import logging

from agent.langgraph_manual import agent, extract_text


# ==================================================
# LOGGING CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Agentic AI Assistant",
    description="Backend API for the Agentic AI Assistant",
    version="1.0"
)


# ==================================================
# REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )

    thread_id: str = Field(
        default="conversation_1",
        min_length=1,
        max_length=100
    )


# ==================================================
# RESPONSE MODEL
# ==================================================

class ChatResponse(BaseModel):

    response: str
    thread_id: str


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():

    logger.info("Root endpoint accessed")

    return {
        "message": "Agentic AI Assistant API is running"
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    # --------------------------------------------------
    # Clean input
    # --------------------------------------------------

    request.message = request.message.strip()
    request.thread_id = request.thread_id.strip()

    # --------------------------------------------------
    # Validate message
    # --------------------------------------------------

    if not request.message:

        logger.warning(
            "Empty message received | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # --------------------------------------------------
    # Log request
    # --------------------------------------------------

    # We intentionally do NOT log the complete user message.
    # User messages may contain sensitive information.

    logger.info(
        "Chat request received | thread_id=%s",
        request.thread_id
    )

    # --------------------------------------------------
    # Conversation configuration
    # --------------------------------------------------

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    # --------------------------------------------------
    # Start execution timer
    # --------------------------------------------------

    start_time = time.time()

    # --------------------------------------------------
    # Run LangGraph agent
    # --------------------------------------------------

    try:

        result = agent.invoke(
            {
                "messages": [
                    ("user", request.message)
                ]
            },
            config=config
        )

        # --------------------------------------------------
        # Extract assistant response
        # --------------------------------------------------

        answer = extract_text(
            result["messages"][-1].content
        )

        # --------------------------------------------------
        # Calculate execution time
        # --------------------------------------------------

        execution_time = time.time() - start_time

        # --------------------------------------------------
        # Log successful response
        # --------------------------------------------------

        logger.info(
            "Agent completed | thread_id=%s | execution_time=%.2fs",
            request.thread_id,
            execution_time
        )

        # --------------------------------------------------
        # Return response
        # --------------------------------------------------

        return ChatResponse(
            response=answer,
            thread_id=request.thread_id
        )

    # --------------------------------------------------
    # Handle agent errors
    # --------------------------------------------------

    except Exception:

        logger.exception(
            "Agent execution failed | thread_id=%s",
            request.thread_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The assistant is temporarily unavailable. "
                "Please try again later."
            )
        )


# ==================================================
# FRONTEND
# ==================================================

app.mount(
    "/frontend",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)