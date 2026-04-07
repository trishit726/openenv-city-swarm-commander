import sys
import os

# Add parent dir to path so we can import environment.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from environment import SwarmEnvironment, SwarmCommand, Observation, Reward

# ── Shared environment instance ──────────────────────────────────────────────
# openenv-core's create_fastapi_app has a schema mismatch with our Observation
# (it expects obs.reward which we don't have). We build the HTTP routes manually.
app = FastAPI(
    title="City Swarm Commander",
    description="OpenEnv drone fleet logistics environment for the Meta PyTorch Hackathon.",
    version="1.0.0",
)

_env = SwarmEnvironment(task="easy")


# ── Request/Response models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task: Optional[str] = None          # allow switching task on reset


class StepResponse(BaseModel):
    observation: dict
    reward: dict
    done: bool
    info: dict


# ── OpenEnv HTTP endpoints (what the hackathon validator checks) ──────────────

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    """Reset the environment. Returns the initial Observation."""
    global _env
    if body.task and body.task in ("easy", "medium", "hard"):
        _env = SwarmEnvironment(task=body.task)
    obs = _env.reset()
    return JSONResponse(content=obs.model_dump(), status_code=200)


@app.post("/step")
def step(command: SwarmCommand):
    """Step the environment with a SwarmCommand. Returns observation, reward, done, info."""
    obs, reward, done, info = _env.step(command)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }, status_code=200)


@app.get("/state")
def state():
    """Returns the current Observation without advancing the simulation."""
    return JSONResponse(content=_env.state().model_dump(), status_code=200)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/schema")
def schema():
    """Returns the action and observation schemas."""
    return {
        "action": SwarmCommand.model_json_schema(),
        "observation": Observation.model_json_schema(),
    }


@app.get("/")
def read_root():
    return {
        "status": "City Swarm Commander is online",
        "docs":   "/docs",
        "openenv": "Ready",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET  /state",
        }
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
