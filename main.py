# General imports
import os
import sys
import numpy as np

# Gymnasium
import gymnasium as gym

import copy

# Sinergym
import sinergym
from src.wrappers import SinergymWrapper
from src.rewards import FangerReward

import torch.nn as nn
from ray.rllib.models.catalog import MODEL_DEFAULTS
from copy import deepcopy
from src.utils import *

from src.model import UncertainPPO

# Function to create the agent
def get_agent(
        env, 
        callback_fn, 
        env_config, 
        eval_env_config, 
        model_config,
        args, 
        planning_model=None):
    
    # Initlaize the default configuration
    config = UncertainPPO.get_default_config() 
    
    # Configuration updates and customization
    config["env"] = env
    config["seed"] = 8765
    config["ppo_framework"] = "torch"
    config["_disable_preprocessor_api"] = True
    config["rollout_fragment_length"] = "auto"
    config["env_config"] = env_config
    config["model"] = MODEL_DEFAULTS
    config["model"] = {
        "fcnet_activation": lambda: nn.Sequential(nn.Tanh(), nn.Dropout(p=0.1)),   # Custom_Activation
        "dropout": 0.1,
        "num_dropout_evals": 10,
        "max_seq_len": 1,
    }
    config["train_batch_size"] = 26280
    config["num_sgd_iter"] = 3
    config["disable_env_checking"] = True
    config["clip_param"] = 0.3
    config["lr"] = 5e-05
    config["gamma"] = 0.8
    config["evaluation_interval"] = 3
    config["evaluation_duration"] = 3 #157680              # rrlib config
    config["evaluation_duration_unit"] = "timesteps"    #"episodes"
    config["horizon"] = args['horizon'],                # rrlib config
    config["soft_horizon"] = True                       # rrlib config        
    config["restart_failed_sub_environments"] = True    # rrlib config
    config["evaluation_sample_timeout_s"] = 3600        # rrlib config
    config["evaluation_parallel_to_training"] = False   # rrlib config
    config["evaluation_config"] = {
        "env_config": eval_env_config
    }
    config["callbacks"] = lambda: callback_fn(
        num_descent_steps=91,
        batch_size=1, 
        no_coop=False, 
        planning_model=planning_model,
        config=config, 
        run_active_rl=1.0, 
        planning_uncertainty_weight=1, 
        args=args, 
        uniform_reset=0.0)
    
    # Disable environment checking
    config["disable_env_checking"] = True
    
    # Initialize your custom or updated RLlib Algorithm
    agent = UncertainPPO(config=config)

    return agent

# Function to train the agent
def train_agent(agent, num_iterations=2):
    # Simple training loop
    for i in range(num_iterations):
        # Perform one iteration of training the policy with PPO
        result = agent.train()
        print(f"ITERATION: {i}, reward: {result['episode_reward_mean']}")
        # Save the model
        agent.save("checkpoint")


if __name__ == "__main__":
    # Add the EnergyPlus path to the system path
    sys.path.append('./EnergyPlus-23-1-0')
    # Set the EnergyPlus path as an environment variable
    os.environ['EPLUS_PATH'] = './EnergyPlus-23-1-0'

    # Environment ID
    id = "Eplus-5zone-hot-continuous-stochastic-v1"

    # Weather
    weather_files = ['USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw']

    # mu, sigma and theta for the weather variability
    # In the original version, weather_variability was a triple (mu, sigma, theta)
    # that only affected the drybulb (outdoor temperature), while now it is a dictionary
    # with the mean, sigma and theta for each weather variable we want to vary.
    # NOTE: Check apply_weather_variability method in CustomModelJSON class
    weather_variability = {
        'drybulb': np.array([5.53173187e+00, 0.00000000e+00, 2.55034944e-03]), 
        'relhum': np.array([1.73128872e+01, 0.00000000e+00, 2.31712760e-03]), 
        'winddir': np.array([7.39984654e+01, 0.00000000e+00, 4.02298013e-04]), 
        'dirnorrad': np.array([3.39506556e+02, 0.00000000e+00, 9.78192172e-04]), 
        'windspd': np.array([1.64655725e+00, 0.00000000e+00, 3.45045547e-04])}

    variability_low = {
        'drybulb': np.array([4.31066896e+00, 1.43882821e-03]), 
        'relhum': np.array([2.07871802e+01, 1.52442626e-03]), 
        'winddir': np.array([9.2461295e+01, 1.7792310e-04]), 
        'dirnorrad': np.array([2.26216882e+02, 3.96634341e-04]), 
        'windspd': np.array([1.92756975e+00, 2.60994514e-04])
    }

    variability_high = {
        'drybulb': np.array([9.87995071e+00, 8.40623734e-03]), 
        'relhum': np.array([3.26129158e+01, 5.10374079e-03]), 
        'winddir': np.array([1.46046002e+02, 5.68863159e-04]), 
        'dirnorrad': np.array([3.51914077e+02, 8.28838542e-04]), 
        'windspd': np.array([3.73801488e+00, 8.64436358e-04])
    }

    # Custom reward derived from Fanger's comfort model.
    # This extends the LinearReward class from sinergym adding ppd and occupancy variables.
    reward = FangerReward
    reward_kwargs={
        'temperature_variables': ['air_temperature'],
        'energy_variables': ['HVAC_electricity_demand_rate'],
        'range_comfort_winter': [20, 23],
        'range_comfort_summer': [23, 26],
        'energy_weight': 0.1,
        'ppd_variable': 'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)',
        'occupancy_variable': 'Zone People Occupant Count(SPACE1-1)'
    }

    # Create the environment
    env =  SinergymWrapper()

    # Define the arguments
    args = {
        'num_gpus': 1, 
        'log_path': 'logs',
        'project': 'active-rl', 
        'profile': False, 
        'env': 'sg', 
        'num_timesteps': 10, # 7500000, 
        'train_batch_size': 16, # 26280, 
        'horizon': 3, #4380, 
        'clip_param': 0.3, 
        'lr': 2e-04, 
        'gamma': 0.8, 
        'num_sgd_iter': 3, 
        'eval_interval': 3, 
        'num_training_workers': 1, 
        'num_eval_workers': 1, 
        'num_envs_per_worker': 1, 
        #'cl_filename': './data/citylearn_challenge_2022_phase_1/schema.json', 
        #'cl_eval_folder': './data/all_buildings', 
        #'cl_use_rbc_residual': 0, 
        #'cl_action_multiplier': 1, 
        #'gw_filename': 'gridworlds/sample_grid.txt', 
        #'gw_steps_per_cell': 1, 
        #'dm_filename': 'gridworlds/sample_grid.txt', 
        'aliveness_reward': 0.01, 
        'distance_reward_scale': 0.01, 
        'use_all_geoms': False, 
        'walker': 'ant', 
        'dm_steps_per_cell': 1, 
        'control_timestep': 0.1, 
        'physics_timestep': 0.02, 
        'use_rbc': 0, 
        'use_random': 0, 
        'only_drybulb': False, 
        'sample_envs': False, 
        'sinergym_timesteps_per_hour': 1,
        'eval_fidelity_ratio': 1, 
        'base_weather': 'hot', 
        'random_month': False, 
        'no_noise': False, 
        'continuous': True, 
        'only_vary_offset': True, 
        'use_activerl': 1.0, 
        'use_random_reset': 0.0, 
        'num_descent_steps': 91, 
        'no_coop': False, 
        'planning_model_ckpt': None, 
        'seed': 8765, 
        'planning_uncertainty_weight': 1, 
        'activerl_lr': 0.01, 
        'activerl_reg_coeff': 0.5, 
        'num_dropout_evals': 10, 
        'plr_d': 0.0, 
        'plr_beta': 0.1, 
        'plr_envs_to_1': 100, 
        'plr_robust': False, 
        'naive_grounding': False, 
        'env_repeat': 1, 
        'start': 0, 
        'plr_rho': 0.1, 
        'dropout': 0.1, 
        'full_eval_interval': 10, 
        'sinergym_sweep': '1.0,0,0,0'
    }

    # Define the environment configuration
    env_config = {
        'weather_variability': weather_variability, 
        'variability_low': variability_low, 
        'variability_high': variability_high, 
        'use_rbc': 0, 
        'use_random': 0, 
        'sample_environments': False, 
        'timesteps_per_hour': 1, 
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw', 
        'epw_data': EPW_Data.load("data/US_epw_OU_data.pkl"), 
        'continuous': True, 
        'random_month': False, 
        'only_vary_offset': True
    }

    # Define the evaluation environment configuration
    eval_env_config = deepcopy(env_config)
    eval_env_config["act_repeat"] = 1

    # Define the model configuration
    model_config = {}
    
    # Create the agent
    agent = get_agent(
        SinergymWrapper,
        SinergymCallback, 
        env_config, 
        eval_env_config, 
        model_config, 
        args)
    
    # Train the agent
    train_agent(agent, num_iterations=20)

    # Close the environment
    env.close()

    # Print the training finished message
    print("\n\nTRAINING FINISHED!\n\n")

