"""
FastAPI app serving the LaundryEnv website and REST API.
This powers the beautiful laundry marketplace web UI.
"""
import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("Install: pip install fastapi uvicorn")
    sys.exit(1)

from env.laundry_env import LaundryEnv
from env.models import Action

app = FastAPI(title="LaundryEnv Marketplace", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (for demo)
sessions: Dict[str, LaundryEnv] = {}


class StartRequest(BaseModel):
    task_id: str = "task_easy"
    session_id: Optional[str] = None


class ActionRequest(BaseModel):
    session_id: str
    action_type: str
    payload: Dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the marketplace website."""
    with open("templates/index.html") as f:
        return f.read()


@app.post("/api/start")
async def start_session(req: StartRequest):
    sid = req.session_id or f"sess_{len(sessions)+1}"
    env = LaundryEnv(task_id=req.task_id)
    obs = env.reset()
    sessions[sid] = env
    return {"session_id": sid, "observation": obs.dict()}


@app.post("/api/step")
async def step(req: ActionRequest):
    env = sessions.get(req.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found. Call /api/start first.")
    try:
        action = Action(action_type=req.action_type, payload=req.payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    obs, reward, done, info = env.step(action)
    result = {"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info}
    if done:
        result["final_grade"] = env.final_grade()
    return result


@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    env = sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state()


@app.get("/api/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "task_easy",   "name": "Order Triage",          "difficulty": "easy",   "description": "Route incoming orders to the correct department."},
            {"id": "task_medium", "name": "Inventory Management",   "difficulty": "medium", "description": "Optimise stock levels and pricing."},
            {"id": "task_hard",   "name": "Customer Support",       "difficulty": "hard",   "description": "Resolve multi-step customer complaints."},
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "env": "LaundryEnv v1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
