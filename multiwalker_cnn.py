import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from pettingzoo.sisl import multiwalker_v9

def train_multiwalker():
    # Create the Multiwalker environment
    env = multiwalker_v9.parallel_env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3,
                                      forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0,
                                      shared_reward=True, terminate_on_fall=True, remove_on_fall=True,
                                      max_cycles=500)

    # Apply wrappers to the environment
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

    # Create and train the PPO model
    model = PPO(CnnPolicy, env, verbose=1, learning_rate=0.0003, n_steps=2048,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)

    model.learn(total_timesteps=2000000)

    # Save the trained model
    model.save("multiwalker_ppo_model_cnn")

    print("Training completed and model saved.")

if __name__ == "__main__":
    train_multiwalker()