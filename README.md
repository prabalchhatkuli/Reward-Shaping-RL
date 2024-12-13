# Reinforcement Learning Reward Shaping Project
This repository contains code and documentation for a project aimed at improving the performance of agents in the Multiwalker environment of PettingZoo using reward shaping. The project introduces custom shaping functions and includes scripts for training, evaluation, and the modification of source code to incorporate these new features.



# Project Structure
1. multiwalker_base.py
    Includes modifications to the original PettingZoo Multiwalker environment. These changes introduce custom reward shaping functions designed to enhance agent coordination and performance.
2. train.py
    Script used to train a reinforcement learning model within the Multiwalker environment using the reward shaping methods.
3. evaluate.py
    Script for evaluating the performance of a trained model. This evaluates how far the package can be carried to, which can also be regarded as a distance between current policy and the optimal policy.



# Requirments
1. PettingZoo
2. Gym
3. stable_baselines3
4. supersuit



# Usage
1. Custom Reward Shaping
    The custom shaping functions are implemented in the modified source code: multiwalker_base.py. Replace this script with original multiwalker_base.py in multiwalker package installed. Ensure the modified file correctly replaces the original file when running other scripts.
2. Training.
    Run the training script to train a model with reward shaping enabled:
    python train.py
3. Evaluation.
    Evaluate the performance of a trained model using:
    python evaluate.py



# Reward Shaping Functions
Four particular functions that we found useful are implemented in multiwalker_base.py. 

1. Reward for package moving(line 551-553):

    package_shaping = self.forward_reward * 130 * self.package.position.x / SCALE
    rewards += package_shaping - self.prev_package_shaping
    self.prev_package_shaping = package_shaping

2. Reward for walker moving(line 507-509):

    walker_forward_shaping = self.walker_forward_reward * 130 * self.walkers[i].hull.position.x / SCALE
    rewards[i] += walker_forward_shaping - self.pre_walker_forward_shaping[i]
    self.pre_walker_forward_shaping[i] = walker_forward_shaping

3. Reward for hull angle keeps horizaontal(line 483-485)

    shaping = -5.0 * abs(walker_obs[0])
    rewards[i] = shaping - self.prev_shaping[i]
    self.prev_shaping[i] = shaping

4. Reward for walker under the package(line 512-519)

    if neighbor_obs[4] < 0.0:
        out_of_package_range_shaping = self.out_of_package_reward * neighbor_obs[4]  # Penalty for moving further out of range behind the package
    elif neighbor_obs[4] > 1.0:
        out_of_package_range_shaping = self.out_of_package_reward * (1 - neighbor_obs[4])  # Penalty for moving further out of range ahead of the package
    else:
        out_of_package_range_shaping = 0  # No penalty/reward if within range
    rewards[i] += out_of_package_range_shaping - self.prev_out_of_package_range_shaping[i]
    self.prev_out_of_package_range_shaping[i] = out_of_package_range_shaping


# Result
1. The result models are within:
    result_models.rar


