import gymnasium as gym
import highway_env
import numpy as np
import random

def heuristic_policy(state):
    # Ego vehicle is at state[0]
    other_vehicles = state[1:]

    #Default action: idle
    action = 1

    for vehicle in other_vehicles:

        pos = vehicle[1]  # Distance ahead (+) or behind (-) relative to ego vehicle
        lane = vehicle[2]  # Lane relative to ego vehicle (-0.666, -0.333, 0, 0.333, 0.666)
        
        # If a car is in the same lane and ahead
        if abs(lane) < 0.05 and pos > 0:
            # Emergency brake if very close
            if pos < 0.15:
                action = 4
                break
            
            # When a car is close, consider also lane change
            elif pos < 0.4:
                # Lane checking
                left_clear = not any(0 < v[1] < 0.2 and -0.4 < v[2] < -0.05 for v in other_vehicles)
                right_clear = not any(0 < v[1] < 0.2 and 0.05 < v[2] < 0.4 for v in other_vehicles)
                
                # Prioritize lane changes over braking
                if right_clear:
                    action = 2
                elif left_clear:
                    action = 0
                else:
                    action = 4
                break
    
    # If no vehicles close ahead in current lane, accelerate
    if not any(abs(v[2]) < 0.05 and 0 < v[1] < 0.4 for v in other_vehicles):
        action = 3
    
    return action

# Set random seeds
base_seed = 0
np.random.seed(base_seed)
random.seed(base_seed)

env = gym.make("highway-v0", config={"lanes_count": 3, "ego_spacing": 1.5, "manual_control": False,  'high_speed_reward': 1}, render_mode='human')

state, _ = env.reset(seed = base_seed)

done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

results = []

while episode <= 10:
    episode_steps += 1
    
    action = heuristic_policy(state)
    state, reward, done, truncated, _ = env.step(action)
    env.render()
    
    episode_return += reward
    
    if done or truncated:
        results.append([episode, episode_steps, episode_return, done])
        print(f"Episode {episode}: Steps={episode_steps}, Return={episode_return:.3f}, Crash={done}")

        episode_seed = base_seed + episode
        state, _ = env.reset(seed=episode_seed)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

np.savetxt("baseline_results.csv", results, delimiter=",", header="Episode,Steps,Return,Crash", comments='')