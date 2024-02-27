import pandas as pd
import numpy as np
import gymnasium as gym
from sinergym.envs.eplus_env import EplusEnv

from src.modeling import CustomModelJSON
from src.rewards import FangerReward

import torch
from gymnasium.spaces.box import Box
from typing import Optional, Union
from sinergym.utils.wrappers import NormalizeObservation

WEATHER_VAR_MAP = {
    'Site Outdoor Air Drybulb Temperature(Environment)': "drybulb",
    'Site Outdoor Air Relative Humidity(Environment)': "relhum",
    'Site Wind Direction(Environment)': "winddir",
    'Site Direct Solar Radiation Rate per Area(Environment)': "dirnorrad",
    'Site Diffuse Solar Radiation Rate per Area(Environment)': "difhorrad",
    'Site Wind Speed(Environment)': "windspd"
}
REVERSE_WEATHER_MAP = {v: k for k, v in WEATHER_VAR_MAP.items()}

class SinergymWrapper(gym.ObservationWrapper):

    def __init__(
            self,
            env: EplusEnv = None,
            env_name: str = 'eplus-env-v1',
            OU_params_path: str = 'data/US_epw_OU_params.csv'):
        
        # Create the environment
        env = gym.make(
            id="Eplus-5zone-hot-continuous-stochastic-v1",
            weather_files=['USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'],
            reward=FangerReward,
            reward_kwargs={
                'temperature_variables': ['air_temperature'],
                'energy_variables': ['HVAC_electricity_demand_rate'],
                'range_comfort_winter': [20, 23],
                'range_comfort_summer': [23, 26],
                'energy_weight': 0.1,
                'ppd_variable': 'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)',
                'occupancy_variable': 'Zone People Occupant Count(SPACE1-1)'
            },
            weather_variability={
                'drybulb': np.array([5.53173187e+00, 0.00000000e+00, 2.55034944e-03]), 
                'relhum': np.array([1.73128872e+01, 0.00000000e+00, 2.31712760e-03]), 
                'winddir': np.array([7.39984654e+01, 0.00000000e+00, 4.02298013e-04]), 
                'dirnorrad': np.array([3.39506556e+02, 0.00000000e+00, 9.78192172e-04]), 
                'windspd': np.array([1.64655725e+00, 0.00000000e+00, 3.45045547e-04])}
        )
        # Set only_vary_offset to True to only vary the offset
        self.only_vary_offset = True

        # Modify the environment model with a custom one
        env.unwrapped.model = CustomModelJSON(
            env_name=env_name,
            json_file=env.unwrapped.building_file,
            weather_files=env.unwrapped.weather_files,
            actuators=env.unwrapped.actuators,
            variables=env.unwrapped.variables,
            meters=env.unwrapped.meters,
            max_ep_store=env.unwrapped.model.max_ep_store,
            extra_config=env.unwrapped.model.config
        )       

        # Normalize the observation space
        env =  NormalizeObservation(env)

        # Call the parent constructor
        super(SinergymWrapper, self).__init__(env)

        # Read the Ornstein-Uhlenbeck (OU) process parameters from a CSV file
        self.OU_params_df = pd.read_csv(OU_params_path)

    
    # TODO: Implement the methods below if necessary
    
    # ok
    def observation(self, observation):
        return observation

    # ok
    def reset(self, seed=None, initial_state: Optional[Union[int, np.ndarray]]=None):   
        """
        Resets the environment with flexible initialization options.
        - If `initial_state` is an int, it specifies a scenario index for predefined weather variabilities,
          or signals to sample new weather variabilities if negative.
        - If `initial_state` is a dict, it directly specifies the weather conditions.
        - If `initial_state` is an ndarray, it represents a specific state to initialize to.
        - If `initial_state` is None, the environment resets to default conditions.
        """
        if isinstance(initial_state, int):
            if initial_state < 0 and self.environment_variability_file is not None:
                weather_variability = self.sample_variability()
                # print("SAMPLED VARIABILITY", curr_weather_variability)
            elif 0 <= initial_state < len(self.weather_variability):
                self.scenario_idx = initial_state
                weather_variability = self.weather_variability[initial_state]
                # print("PRESET VARIABILITY", weather_variability)
            else:
                raise ValueError("Initial state does not specify a valid weather variability.")
            obs, info =  self.env.reset(weather_variability=weather_variability)

        elif isinstance(initial_state, dict):
            obs, info = self.env.unwrapped.reset(weather_variability=initial_state)

        elif isinstance(initial_state, np.ndarray):
            # If we only vary the offset, the variability params are actually not in the provided state
            if self.only_vary_offset:
                # Use the default variability params
                variability = np.zeros([len(self.active_variables) * 2])
                # Iterate through the active variables and set the variability params
                for i, (key, variability) in enumerate(self.env.unwrapped.weather_variability.items()):
                    variability[2 * i] = variability[0]
                    variability[2 * i + 1] = variability[2]
            else:
                # Extract the variability params from the state 
                variability = initial_state[..., -self.num_extra_variability_dims:]
                # Clip the variability to the specified range
                variability = variability * self.variability_scale + self.variability_offset
                variability = np.clip(variability, self.variability_low, self.variability_high)

            # Create a dictionary to pass to the environment
            variability_dict = {}

            # Iterate through the active variables and set the variability params
            for var_name in self.active_variables:
                # Get the indices of the variability params for the current variable
                idxs = self.variability_noise_idxs[var_name]
                idxs = [idx - self.original_obs_space_shape[-1] for idx in idxs]
                # Get the variability params for the current variable
                variability_params = variability[idxs]
                # Get the offset from the initial state
                offset_idx = self.variability_offset_idxs[var_name]
                offset = np.clip(initial_state[..., offset_idx], 0, 1)
                var_range = self._get_range(var_name)
                offset = offset * (var_range[1] - var_range[0]) + var_range[0]
                #offset -= base_weather[var_name]
                # Set the variability params in the dictionary
                variability_dict[var_name] = (variability_params[0], offset, variability_params[1])
            # print("ACTIVE VARIABILITY", variability_dict)
            self.last_variability = variability_dict
            obs, info = self.env.unwrapped.reset(weather_variability=variability_dict)
        
        else:
            obs, info = self.env.reset()
        
        return obs #, info

    # ok
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info

    # ok
    def close(self):
        self.env.close()
    
    

    # TODO: check correctness   ##########################################################################################################################
    
    # copyed
    def separate_resettable_part(self, obs):
        """
        Separates the observation into the resettable portion and the original. 
        Make sure this operation is differentiable
        """
        resettable = []

        # Iterate over the weather variability keys
        for key in self.active_variables:
            offset = torch.unsqueeze(obs[..., self.variability_offset_idxs[key]], dim=-1)
            resettable.append(offset)
        # 
        if not self.only_vary_offset:
            resettable.append(obs[..., -self.num_extra_variability_dims:])
        resettable = torch.concat(resettable, dim=-1)
        return resettable, obs

    # copyed
    def combine_resettable_part(self, obs, resettable):
        """
        Combines an observation that has been split like in separate_resettable_part back together. 
        Make sure this operation is differentiable
        """
        obs = obs.detach()
        
        for i, key in enumerate(self.active_variables):
            obs[..., self.variability_offset_idxs[key]] = resettable[..., i]

        if not self.only_vary_offset:
            obs[..., -self.num_extra_variability_dims:] = resettable[..., len(self.active_variables):]
        return obs

    # copyed
    def sample_obs(self, naive_grounding=False):
        """
        Automatically sample an observation to seed state generation
        """
        obs = self.start_obs
        if naive_grounding:
            weather_df = self.env.unwrapped.simulator._config.weather_data.get_weather_series()
            weather_means = weather_df.mean(axis=0)
            base_weather = weather_df.iloc[0] if not self.offset_by_means else weather_means
            variability = self.sample_variability()
            for i, var_name in enumerate(self.active_variables):
                offset_idx = self.variability_offset_idxs[var_name]
                var_range = self._get_range(var_name)
                offset = variability[var_name][1] + base_weather[var_name] # use index one because that's where the offset is.
                offset = (offset - var_range[0]) / (var_range[1] - var_range[0])
                obs[offset_idx] = offset
                
        return obs
    
    # copyed
    def resettable_bounds(self):
        """
        Get bounds for resettable part of observation space
        """
        reset_dims = len(self.active_variables)
        if not self.only_vary_offset:
            reset_dims += self.num_extra_variability_dims
        low = np.zeros([reset_dims])
        high = np.ones([reset_dims])
        return low, high
    
    #######################################################################################################################################################

    # ok
    def sample_variability(self):
        """
        Samples a row from a dataframe containing Ornstein-Uhlenbeck (OU) process parameters for 
        various weather variables.
        """
        # Sample a row and directly access it
        row = self.OU_params_df.sample(1).iloc[0]
        # Return a dictionary with the sampled values
        return {variable: [row[f"{variable}_{j}"] for j in range(3)] for variable in self.weather_variability}

    # copyed
    def _augment_obs_space(self, env, config):
        obs_space = env.unwrapped.observation_space
        self.original_obs_space_shape = obs_space.shape
        self.active_variables = list(self.env.unwrapped.weather_variability.keys()) 
        #self.variability_offset_idxs = {WEATHER_VAR_MAP.get(variable, ""): i for i, variable in enumerate(self.env.unwrapped.variables["observation"])}
        self.last_untransformed_obs = None
        
        obs_space_shape_list = list(obs_space.shape)
        i = obs_space_shape_list[-1]

        # Maps weather variable name to indexes in observation
        # Has entries for both the epw format (e.g. 'drybulb') and the sinergym format
        # (e.g. 'Site Outdoor Air Drybulb Temperature(Environment)')
        self.variability_noise_idxs = {} 
        self.variability_low = []
        self.variability_high = []
        for key, variability in self.env.unwrapped.weather_variability.items():
            self.variability_noise_idxs[key] = list(range(i, i+2))
            if "variability_low" in config:
                self.variability_low.extend(config["variability_low"][key])
            else:
                self.variability_low.extend(list(variability))
            
            if "variability_high" in config:
                self.variability_high.extend(config["variability_high"][key])
            else:
                self.variability_high.extend(list(variability))
            i += 2
        obs_space_shape_list[-1] = i

        self.variability_low = np.array(self.variability_low)
        self.variability_high = np.array(self.variability_high)

        self.num_extra_variability_dims = len(self.variability_low)
        self.variability_offset = self.variability_low
        self.variability_scale = (self.variability_high - self.variability_low)
        
        # If only the offset variation is considered (only_vary_offset), 
        # the original observation space remains unchanged.
        # Otherwise, the function expands the observation space to account
        # for these variability factors, allowing the agent to receive 
        # detailed environmental conditions as part of its input.
        if self.only_vary_offset:
            self.observation_space = obs_space
            self.last_untransformed_obs = None
            self.num_extra_variability_dims = 0
        else:
            low = list(obs_space.low) + [0. for _ in range(self.num_extra_variability_dims)]
            high = list(obs_space.high) + [1. for _ in range(self.num_extra_variability_dims)]
            self.observation_space = Box(
                low = np.array(low), 
                high = np.array(high),
                shape = obs_space_shape_list,
                dtype=np.float32)
    