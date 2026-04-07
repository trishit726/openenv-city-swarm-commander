import os
import json
from openai import OpenAI
from environment import SwarmEnvironment, SwarmCommand

def run_baseline():
    """
    Runs the baseline evaluation for City Swarm Commander.
    Uses the OpenAI client to query an LLM for agentic actions based on state.
    """
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    
    # Read token (OpenAI standard or HF standard for local endpoints)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "dummy_key"

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    tasks = ["easy", "medium", "hard"]
    
    print("Initializing City Swarm Commander Baseline Evaluation...\n")
    
    for task in tasks:
        print(f"[START] Task: {task}")
        env = SwarmEnvironment(task=task)
        state = env.state()
        done = False
        
        while not done:
            # We construct a prompt for the model
            syst_prompt = (
                "You are the Swarm Commander managing a drone fleet. "
                "Output valid JSON for SwarmCommand. "
                "Available actions: 'assign_delivery', 'recharge_drone', 'no_op'. "
                "Consider battery drain (higher in rain/storm) and recharge at (6,6) when low."
            )
            user_prompt = f"Current state: {state.model_dump_json()}"
            
            try:
                # Actual LLM call if credentials are valid
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": syst_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                resp_json = json.loads(response.choices[0].message.content)
                command = SwarmCommand(**resp_json)
            except Exception as e:
                # Fallback heuristic logic if API fails
                pending = [d for d in state.deliveries if d.status == "pending"]
                idle_drones = [d for d in state.drones if d.status == "idle" and d.battery > 20]
                
                if False: # Removed emergencies fallback for now since emergency array in state changed
                    pass
                elif pending and idle_drones:
                    command = SwarmCommand(
                        action_type="assign_delivery",
                        drone_id=idle_drones[0].id,
                        target_id=pending[0].id
                    )
                else:
                    command = SwarmCommand(action_type="no_op")
            
            print(f"[STEP] {state.time_step} | Action: {command.model_dump_json()}")
            state, reward, done, _ = env.step(command)
            
        print(f"[END] Task: {task} | Final Score: {state.current_mission_score}\n")

if __name__ == "__main__":
    run_baseline()
