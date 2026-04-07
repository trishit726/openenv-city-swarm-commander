import sys
import os

# Add parent dir to path so we can import environment.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from openenv.core.env_server import create_fastapi_app
from environment import SwarmEnvironment, SwarmCommand, Observation

# Pure OpenEnv FastAPI app — no Gradio UI
app = create_fastapi_app(lambda: SwarmEnvironment(task="easy"), SwarmCommand, Observation)


@app.get("/")
def read_root():
    return {
        "status": "City Swarm Commander is online",
        "docs":   "/docs",
        "openenv": "Ready",
        "tasks": ["easy", "medium", "hard"],
    }


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
