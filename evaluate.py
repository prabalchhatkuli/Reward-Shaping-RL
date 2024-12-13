import supersuit as ss
from stable_baselines3 import PPO
from pettingzoo.sisl import multiwalker_v9
from gymnasium.wrappers import RecordVideo
import numpy as np

def test_multiwalker():
    # Create the environment with rendering enabled
    env = multiwalker_v9.parallel_env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3,
                                      forward_reward=1.0, walker_forward_reward=0.0,
                                      out_of_package_reward = 0.0, location_reward = 0.0,
                                      terminate_reward=0.0, fall_reward=0.0,
                                      shared_reward=True, terminate_on_fall=True, remove_on_fall=True,
                                      terrain_length=1000, max_cycles=100000, render_mode="human")
    
    # Apply the same wrappers as during training
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')

    # Load the trained model
    model = PPO.load("multiwalker_ppo_model")

    # Reset the environment
    obs = env.reset()

    total_rewards = []
    num_episodes = 50  # Number of episodes to test

    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = []
        done = False
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)
            # Take a step in the environment
            obs, rewards, dones, infos = env.step(action)
            # Record rewards
            episode_rewards.append(rewards)
            # Check if the episode is done
            done = np.all(dones)
            # Render the environment
            env.render()
        
        # Calculate total reward for the episode
        total_reward = np.sum(episode_rewards)
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Close the environment
    env.close()

    # Calculate average reward over all episodes
    average_reward = np.mean(total_rewards)
    print(f"Average Total Reward over {num_episodes} episodes: {average_reward}")

if __name__ == "__main__":
    test_multiwalker()

    
