import gymnasium
import highway_env
import random
import numpy as np

base_seed = 0
np.random.seed(base_seed)
random.seed(base_seed)

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5,  'high_speed_reward': 1},
                     render_mode='human')

env.reset(seed = base_seed)

done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

results = []

while episode <= 10:
    episode_steps += 1

    _, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

    if done or truncated:
        results.append([episode, episode_steps, episode_return, done])
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        episode_seed = base_seed + episode
        env.reset(seed = episode_seed)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

np.savetxt("manual_control_results.csv", results, delimiter=",", header="Episode,Steps,Return,Crash", comments='')