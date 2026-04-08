from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional, Tuple, List, Dict
import numpy as np
import copy
import random

# ==========================================
# 1. Action Model (Input from LLM)
# ==========================================

class SwarmCommand(BaseModel):
    """
    The high-level command issued by the Swarm Commander (LLM agent).
    Only ONE action can be taken per step.
    """
    action_type: Literal[
        "assign_delivery", "reroute_drone", "recall_drone", "prioritize_emergency",
        "deploy_weather_sensor", "recharge_drone", "query_telemetry", "no_op"
    ] = Field(
        ..., 
        description="The specific tool the Swarm Commander wants to execute. Choose exactly one."
    )

    drone_id: Optional[str] = Field(
        None, 
        description="ID of the drone to control (e.g., 'D1', 'D3'). Required for assign_delivery, reroute_drone, recall_drone, recharge_drone."
    )

    target_id: Optional[str] = Field(
        None, 
        description="ID of the delivery target/package (e.g., 'P5'). Required only for assign_delivery."
    )

    new_waypoint: Optional[Tuple[int, int]] = Field(
        None, 
        description="New grid coordinates (x, y) to reroute to. Example: (7, 4). Required only for reroute_drone."
    )

    delivery_id: Optional[str] = Field(
        None, 
        description="ID of the delivery to mark as high-priority. Required only for prioritize_emergency."
    )

    waypoint: Optional[Tuple[int, int]] = Field(
        None, 
        description="Grid coordinates to scan for weather data. Required only for deploy_weather_sensor."
    )

    model_config = ConfigDict(extra="forbid")

# ==========================================
# 2. Observation Models (Output to LLM)
# ==========================================

class DroneState(BaseModel):
    id: str
    position: Tuple[int, int]
    battery: float
    status: str  # idle, moving, recalling, charging, failed
    cargo: Optional[str]

class DeliveryState(BaseModel):
    id: str
    target_position: Tuple[int, int]
    priority: str  # normal, emergency
    status: str    # pending, assigned, complete, failed

class Emergency(BaseModel):
    id: str
    position: Tuple[int, int]
    severity: str  # high, medium, low

class Observation(BaseModel):
    time_step: int
    drones: List[DroneState]
    deliveries: List[DeliveryState]
    emergencies: List[Emergency]
    weather_condition: str
    weather_affected_areas: List[Tuple[int, int]]
    current_mission_score: float
    natural_language_summary: str

# ==========================================
# 3. Reward Model
# ==========================================

class Reward(BaseModel):
    step_reward: float
    breakdown: Dict[str, float]

# ==========================================
# 4. Main Environment Engine Skeleton
# ==========================================

class SwarmEnvironment:
    """
    The SwarmEnvironment simulates a real-world drone fleet logistics command center.
    It restricts the agent strictly to high-level strategic orchestration without 
    involving continuous/low-level physics engines.
    """
    def __init__(self, task: str = "easy"):
        self.task = task
        self.grid_size = 12
        self.base_station = (6, 6)
        self.max_steps = 40
        self.reset()
    
    def reset(self, **kwargs):
        """Resets the environment back to step 0 according to the selected Task."""
        self.time_step = 0
        self.current_mission_score = 0.01
        self.weather_condition = "clear"
        self.weather_affected_areas = []
        self.drones = []
        self.deliveries = []
        self.emergencies = []
        
        # Difficulty configuration
        if self.task == "easy":
            num_drones, num_deliveries = 4, 6
        elif self.task == "medium":
            num_drones, num_deliveries = 6, 12
            self.weather_condition = "rain"
            self.weather_affected_areas = [(x, y) for x in range(3, 9) for y in range(3, 9)]
        elif self.task == "hard":
            num_drones, num_deliveries = 8, 20
            self.weather_condition = "storm"
            self.weather_affected_areas = [(x, y) for x in range(12) for y in range(12)]
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        for i in range(num_drones):
            self.drones.append(DroneState(
                id=f"D{i+1}", 
                position=self.base_station, 
                battery=100.0, 
                status="idle",
                cargo=None
            ))
        
        for i in range(num_deliveries):
            loc = (random.randint(0, 11), random.randint(0, 11))
            while loc == self.base_station:
                loc = (random.randint(0, 11), random.randint(0, 11))
            self.deliveries.append(DeliveryState(
                id=f"P{i+1}", 
                target_position=loc, 
                priority="normal",
                status="pending"
            ))
            
        return self.state()

    def state(self) -> Observation:
        """Returns the completely strongly-typed Observation state to the LLM agent."""
        pending_count = sum(1 for d in self.deliveries if d.status == 'pending')
        idle_count = sum(1 for d in self.drones if d.status == 'idle')
        
        summary = (
            f"Step {self.time_step}: Weather is {self.weather_condition}. "
            f"There are {pending_count} deliveries pending and {idle_count} drones currently idle. "
            f"Current mission score is an established {self.current_mission_score:.3f}."
        )
        
        return Observation(
            time_step=self.time_step,
            drones=[copy.deepcopy(d) for d in self.drones],
            deliveries=[copy.deepcopy(d) for d in self.deliveries],
            emergencies=[copy.deepcopy(e) for e in self.emergencies],
            weather_condition=self.weather_condition,
            weather_affected_areas=copy.deepcopy(self.weather_affected_areas),
            current_mission_score=self.current_mission_score,
            natural_language_summary=summary
        )

    def step(self, command: SwarmCommand) -> tuple[Observation, Reward, bool, dict]:
        """Steps the simulation forward strictly using exactly one valid action_type mapping."""
        self.time_step += 1
        reward_breakdown = {"movement": 0.0, "completion": 0.0, "emergency": 0.0, "penalty": 0.0}
        
        drone_map = {d.id: d for d in self.drones}
        delivery_map = {d.id: d for d in self.deliveries}
        
        # Environment Physics Tick (based on status set in PREVIOUS step)
        for d in self.drones:
            # Handle Charging
            if d.status == "charging":
                d.battery = min(100.0, d.battery + 15.0)
                if d.battery >= 100.0:
                    d.status = "idle"
                continue

            # Battery Drain Logic based on Weather
            drain_rate = 1.0
            if self.weather_condition == "rain":
                drain_rate = 2.5
            elif self.weather_condition == "storm":
                drain_rate = 5.0
            
            if d.status != "idle":
                d.battery -= drain_rate
            else:
                d.battery -= 0.1 # Small idle drain

            # Failure Check
            if d.battery <= 0:
                d.battery = 0
                d.status = "failed"
                if d.cargo:
                    dlv = delivery_map.get(d.cargo)
                    dlv.status = "failed"
                continue

            # Movement Logic
            if d.status == "moving" and d.cargo:
                dlv = delivery_map.get(d.cargo)
                target = dlv.target_position
                
                if d.position[0] < target[0]:
                    d.position = (d.position[0] + 1, d.position[1])
                elif d.position[0] > target[0]:
                    d.position = (d.position[0] - 1, d.position[1])
                elif d.position[1] < target[1]:
                    d.position = (d.position[0], d.position[1] + 1)
                elif d.position[1] > target[1]:
                    d.position = (d.position[0], d.position[1] - 1)
                
                reward_breakdown["movement"] += 0.02
                
                if d.position == target:
                    dlv.status = "complete"
                    d.cargo = None
                    d.status = "idle" # Returns to idle at current pos, agent should recall if needed
                    reward_breakdown["completion"] += 0.5

        # Action Resolution Block (sets status for NEXT step)
        if command.action_type == "assign_delivery" and command.drone_id and command.target_id:
            d = drone_map.get(command.drone_id)
            dlv = delivery_map.get(command.target_id)
            if d and dlv and d.status == "idle" and dlv.status == "pending" and d.battery > 10:
                d.status = "moving"
                d.cargo = dlv.id
                dlv.status = "assigned"
            else:
                reward_breakdown["penalty"] -= 0.05
        
        elif command.action_type == "recharge_drone" and command.drone_id:
            d = drone_map.get(command.drone_id)
            if d and d.position == self.base_station:
                d.status = "charging"
            else:
                reward_breakdown["penalty"] -= 0.05

        elif command.action_type == "no_op":
            pass
        else:
            reward_breakdown["penalty"] -= 0.01

        # Reward normalizer (internal reward tracking)
        step_reward = sum(reward_breakdown.values())
        
        # Effective Binary scoring: 0.99 if ALL deliveries complete, 0.01 otherwise
        # This satisfies strictly (0, 1) range while remaining binary.
        all_complete = all(dlv.status == "complete" for dlv in self.deliveries)
        self.current_mission_score = 0.99 if all_complete else 0.01
        
        all_done = all(dlv.status in ["complete", "failed"] for dlv in self.deliveries)
        done = bool(all_done or self.time_step >= self.max_steps)

        rwd = Reward(step_reward=round(step_reward, 3), breakdown=reward_breakdown)
        
        return self.state(), rwd, done, {}

    def close(self):
        """Optional cleanup method for OpenEnv compatibility."""
        pass

    async def reset_async(self, **kwargs):
        return self.reset(**kwargs)
        
    async def step_async(self, command: SwarmCommand):
        return self.step(command)
        
    async def close_async(self):
        return self.close()

    def render(self, mode: str = "human") -> None:
        """
        Optional rendering method for debugging or generating visual assets.
        This method is purely visual and does not affect core environment physics or scoring.
        It uses matplotlib to display a 12x12 map with drones, deliveries, and battery levels.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib is required for rendering. Install it with `pip install matplotlib`.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Configure Weather Background
        weather_colors = {"clear": "#e6f9ff", "rain": "#cccccc", "storm": "#8c8c8c"}
        ax.set_facecolor(weather_colors.get(self.weather_condition, "#ffffff"))

        # Draw Base Station
        ax.plot(self.base_station[0], self.base_station[1], marker="s", color="black", markersize=12, label="Base Station")
        
        # Draw Deliveries
        for dlv in self.deliveries:
            if dlv.status == "pending":
                color, marker = "blue", "P"
            elif dlv.status == "assigned":
                color, marker = "orange", "P"
            elif dlv.status == "complete":
                color, marker = "green", "d"
            else: # failed
                color, marker = "red", "x"
                
            ax.plot(dlv.target_position[0], dlv.target_position[1], marker=marker, color=color, markersize=8, alpha=0.7)
            ax.text(dlv.target_position[0], dlv.target_position[1]+0.2, dlv.id, fontsize=8, ha="center")

        # Draw Drones
        for d in self.drones:
            # Color by status/battery
            if d.status == "failed":
                color = "black"
            else:
                # Color gradient for battery (Red to Green)
                green_val = max(0, min(1, d.battery / 100.0))
                red_val = 1.0 - green_val
                color = (red_val, green_val, 0)
                
            ax.plot(d.position[0], d.position[1], marker="o", color=color, markersize=10, markeredgecolor="black")
            ax.text(d.position[0], d.position[1]-0.3, f"{d.id}\n({int(d.battery)}%)", fontsize=8, ha="center")

        # Formatting
        ax.set_xlim(-1, self.grid_size)
        ax.set_ylim(-1, self.grid_size)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"City Swarm Commander - Task: {self.task.upper()} | Step: {self.time_step} | Weather: {self.weather_condition}", fontsize=14)
        
        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)
            return rgba

