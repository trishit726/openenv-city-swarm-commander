import os
import json
import asyncio
from typing import List, Optional
from openai import OpenAI
from environment import SwarmEnvironment, SwarmCommand

# --- Environment Configuration (matches hackathon checklist) ---
# Defaults ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")          # No default — must be set as HF Space secret
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional: used when testing from_docker_image()
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "city-swarm-commander"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(task_type: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SwarmEnvironment(task=task_type)
    
    log_start(task=task_type, env=BENCHMARK, model=MODEL_NAME)
    
    state = env.state()
    done = False
    steps_taken = 0
    rewards = []
    final_score = 0.0  # Initialize before try so finally block never hits UnboundLocalError
    success = False
    
    try:
        while not done:
            steps_taken += 1
            
            # Construct Prompt
            syst_prompt = (
                "You are the Swarm Commander. Output valid JSON for SwarmCommand. "
                "Actions: 'assign_delivery', 'recharge_drone', 'no_op'. "
                "Include 'action_type', 'drone_id', and 'target_id'."
            )
            user_prompt = f"State: {state.model_dump_json()}"
            
            error_msg = None
            try:
                # LLM Call
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
            except Exception as e:
                # Fallback to no-op on error
                command = SwarmCommand(action_type="no_op")
                error_msg = str(e)[:50]

            # Step Environment
            state, reward_obj, done, _ = env.step(command)
            
            step_reward = reward_obj.step_reward
            rewards.append(step_reward)
            
            log_step(
                step=steps_taken, 
                action=command.action_type, 
                reward=step_reward, 
                done=done, 
                error=error_msg
            )

        # Calculate final metrics
        final_score = state.current_mission_score
        success = final_score >= 0.99  # 0.99 is success, 0.01 is failure
        
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    run_task(TASK_NAME)
