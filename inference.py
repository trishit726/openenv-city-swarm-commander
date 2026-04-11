import os
import json
import sys
from openai import OpenAI
from environment import SwarmEnvironment, SwarmCommand

# 1. Read environment variables (Hugging Face router for evaluation)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Guidelines: HF_TOKEN Description: Hugging Face API token Requirement: Mandatory (no default required)
    print("Error: HF_TOKEN environment variable is required for inference.py", file=sys.stderr)
    sys.exit(1)

# 2. Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference():
    """
    Main evaluation loop compliant with OpenEnv RL Challenge guidelines.
    Emits regulated [START], [STEP], and [END] logs to stdout.
    """
    tasks = ["easy", "medium", "hard"]
    benchmark_name = "city-swarm-commander"

    for task in tasks:
        env = SwarmEnvironment(task=task)
        state = env.reset()
        
        # [START] task=<task_name> env=<benchmark> model=<model_name>
        print(f"[START] task={task} env={benchmark_name} model={MODEL_NAME}")
        
        done = False
        step_count = 0
        rewards_list = []
        last_error = "null"

        try:
            while not done:
                step_count += 1
                
                # Construct prompt for the agent
                syst_prompt = (
                    "You are the Swarm Commander managing a drone fleet. "
                    "Output valid JSON for SwarmCommand. "
                    "Available actions: 'assign_delivery', 'recharge_drone', 'no_op'. "
                    "Example: {'action_type': 'assign_delivery', 'drone_id': 'D1', 'target_id': 'P1'}"
                )
                user_prompt = f"Current state: {state.model_dump_json()}"
                
                action_repr = "no_op"
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": syst_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    resp_json = json.loads(response.choices[0].message.content)
                    command = SwarmCommand(**resp_json)
                    
                    # Create a string representation for the logs
                    params = []
                    if command.drone_id: params.append(command.drone_id)
                    if command.target_id: params.append(command.target_id)
                    action_repr = f"{command.action_type}"
                    if params:
                        action_repr += f"({', '.join(params)})"
                    
                    last_error = "null"
                except Exception as e:
                    # Fallback to no-op if LLM fails or returns invalid JSON
                    command = SwarmCommand(action_type="no_op")
                    action_repr = "no_op"
                    last_error = str(e).replace("\n", " ").replace("=", ":")

                # Execute step in environment
                state, reward_obj, done, info = env.step(command)
                
                step_reward = float(reward_obj.step_reward)
                rewards_list.append(step_reward)
                
                # [STEP] step=<n> action=<action_repr> reward=<0.00> done=<true|false> error=<msg|null>
                reward_fmt = f"{step_reward:.2f}"
                done_bool_str = "true" if done else "false"
                print(f"[STEP] step={step_count} action={action_repr} reward={reward_fmt} done={done_bool_str} error={last_error}")

        except Exception:
            # Capturing error for the END line if needed, but the loop broke
            pass
        finally:
            env.close()

        # [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
        success_bool_str = "true" if (state and state.current_mission_score >= 1.0) else "false"
        rewards_joined = ",".join([f"{r:.2f}" for r in rewards_list]) if rewards_list else "0.00"
        print(f"[END] success={success_bool_str} steps={step_count} rewards={rewards_joined}")

if __name__ == "__main__":
    run_inference()
