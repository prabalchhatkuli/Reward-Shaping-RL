import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy  # Use MlpPolicy for vector observations
from pettingzoo.sisl import multiwalker_v9

def linear_schedule(initial_lr):
    """
    Linear learning rate schedule.
    """
    def schedule(progress_remaining):
        return initial_lr * progress_remaining
    return schedule

def exponential_decay_schedule(initial_lr, decay_rate=0.4):
    """
    Exponential learning rate schedule.
    """
    def schedule(progress_remaining):
        return initial_lr * (decay_rate ** (1 - progress_remaining))
    return schedule

def train_multiwalker():
    # Create the Multiwalker environment
    env = multiwalker_v9.parallel_env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3, terrain_length=75,
                                      forward_reward=3.0, terminate_reward=-100.0, fall_reward=-10.0,
                                      shared_reward=True, terminate_on_fall=True, remove_on_fall=True,
                                      max_cycles=500)

    # Convert to vectorized environment compatible with stable-baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=8, base_class='stable_baselines3')

    # Define the initial learning rate
    initial_lr = 3e-4

    # Create and train the PPO model
    model = PPO(MlpPolicy, env, verbose=1, learning_rate=initial_lr, n_steps=2048,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                tensorboard_log="./ppo_multiwalker_tensorboard/", device="cuda")

    model.learn(total_timesteps=10000000)

    # Save the trained model
    model.save("multiwalker_ppo_model")

    print("Training completed and model saved.")

if __name__ == "__main__":
    train_multiwalker()