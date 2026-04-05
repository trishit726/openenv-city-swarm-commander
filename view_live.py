import matplotlib.pyplot as plt
from environment import SwarmEnvironment, SwarmCommand

def main():
    print("Launching Live Environment UI Viewer...")
    env = SwarmEnvironment(task="medium")
    
    # Scatter some drones artificially so it looks populated
    try:
        # Move drone 0
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D1", target_id="P1"))
        # Move drone 1
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D2", target_id="P2"))
        # Move drone 2
        env.step(SwarmCommand(action_type="assign_delivery", drone_id="D3", target_id="P3"))
        
        for _ in range(6):
            env.step(SwarmCommand(action_type="no_op"))
            
        print("Opening interactive window on your desktop...")
        # mode="human" pops up the matplotlib GUI window natively instead of returning an image buffer.
        env.render(mode="human")
        
    except Exception as e:
        print(f"Failed to view live UI: {e}")

if __name__ == "__main__":
    main()
