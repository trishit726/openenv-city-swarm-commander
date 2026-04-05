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
            syst_prompt = "You are the Swarm Commander. You must output a valid JSON containing 'action', 'drone_id', and 'target_delivery_id'."
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
                # Fallback heuristic logic if the API call fails or no key provided
                # Greedy assignment
                pending = [d for d in state.deliveries if d.status == "pending"]
                idle_drones = [d for d in state.drones if d.status == "idle" and d.battery > 20]
                
                if env.emergencies:
                    command = SwarmCommand(action="prioritize_emergency")
                elif pending and idle_drones:
                    command = SwarmCommand(
                        action="assign_delivery",
                        drone_id=idle_drones[0].id,
                        target_delivery_id=pending[0].id
                    )
                else:
                    command = SwarmCommand(action="no_op")
            
            print(f"[STEP] {state.step_count} | Action: {command.model_dump_json()}")
            state, reward, done, _ = env.step(command)
            
        print(f"[END] Task: {task} | Final Score: {reward.cumulative_score}\n")

if __name__ == "__main__":
    run_baseline()
