import os
import matplotlib.pyplot as plt
from environment import SwarmEnvironment, SwarmCommand

def main():
    print("Initializing environment to generate screenshot...")
    # Using the medium task (includes rain and 6 drones)
    env = SwarmEnvironment(task="medium")
    
    # We'll artificially scatter the drones to see the movement and battery states
    try:
        # Move drone 0 (drone_id is D1, targets are P1)
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        # Move drone 1
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D2", target_id="P2"))
        # Move drone 2
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D3", target_id="P3"))
        
        # Step forward a few ticks to drain battery slightly and alter positions
        for _ in range(6):
            env.step(SwarmCommand(action_type="no_op"))
            
        print("Rendering RGB Array...")
        img = env.render(mode="rgb_array")
        
        filepath = os.path.join("docs", "screenshot.png")
        plt.imsave(filepath, img)
        print(f"Success! Saved visualization to {filepath}")
        
    except Exception as e:
        print(f"Failed to generate screenshot: {e}")

if __name__ == "__main__":
    main()
