# City Swarm Commander

The environment simulates a real-world drone fleet logistics command center (exactly like Zipline, Wing, Amazon Prime Air, and Skye Air in 2026). The LLM agent acts as the "Swarm Commander" and must maximize the Mission Success Score (0.0-1.0) by completing time-sensitive deliveries while managing battery, weather emergencies, and drone failures. It is NOT low-level physics control -- it is high-level strategic orchestration.

## Motivation
With the massive expansion of urban drone delivery networks in 2026 by market leaders like Zipline and Wing, human operators can no longer orchestrate hundreds of concurrent flights manually. The "Swarm Commander" agent must rely on strategic high-level control to resolve conflicts, prioritize emergencies, and handle volatile weather conditions efficiently.

## Action Space
The agent responds with a JSON `SwarmCommand` containing:
- `action`: String identifying the strategic choice (e.g., `assign_delivery`, `recharge_drone`, `prioritize_emergency`, `no_op`).
- `drone_id`: Target drone for the command.
- `target_delivery_id`: Target delivery to associate with the drone.

## Observation Space
The state is represented as a JSON encompassing:
- `step_count`: Current step of the mission.
- `weather`: Current conditions affecting power drain (`clear`, `rain`, `storm`).
- `drones`: Array of active drone states (location, battery, status).
- `deliveries`: Array of pending and completed requests (target location, deadline, status).
- `emergencies`: Array of active network incidents that degrade performance if unhandled.

## Task Difficulty

| Task | Drones | Deliveries | Modifiers |
|---|---|---|---|
| **Easy** | 4 | 6 | Clear Weather |
| **Medium** | 6 | 12 | Rain (Increased battery drain) |
| **Hard** | 8 | 20 | Storm + High Risk of Drone Failures |

## Baseline Scores
Using the heuristic heuristic matching greedy approach:
- **Easy**: 0.95
- **Medium**: 0.70
- **Hard**: 0.45

## Setup & Running
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Run baseline inference:
```bash
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-key-here"
python inference.py
```
Or use standard OpenEnv `docker build` and run.

## Visualization & Debugging (Optional)
While the core evaluation must remain fully headless and programmatic for judging, `SwarmEnvironment` includes an **optional** matplotlib-based `render()` method to help you visualize the state during development or to generate impressive screenshots for your documentation.

**Important:** The `render()` method does not mutate environment state or affect scoring in any way. It explicitly avoids loading `matplotlib` unless the method is called.

**How to generate screenshots:**
Using the `render` function locally will spin up a 12x12 grid showing drone battery gradients (green->red), moving targets, base station nodes, and weather conditions.
```python
# During testing loops
env.render(mode="human") # Opens a live interactive plot

# For saving high-quality README assets
img_array = env.render(mode="rgb_array")
import matplotlib.pyplot as plt
plt.imsave("docs/screenshot.png", img_array)
```
Ensure you have installed matplotlib to use this feature: `pip install matplotlib`
```bash
docker build -t city-swarm-commander .
docker run city-swarm-commander
```
