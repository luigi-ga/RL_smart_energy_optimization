import json
import sinergym
import numpy as np
import gymnasium as gym
from src.modeling import CustomModelJSON
from sinergym.utils.wrappers import NormalizeObservation

class SinergymWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env_config: dict
        ):
        # Create the environment
        self.env = gym.make(
            id=env_config['env_id'],
            weather_files=env_config['weather_files'],
            reward=env_config['reward'],
            reward_kwargs=env_config['reward_kwargs'],
            weather_variability=env_config['weather_variability'],
        )     

        # Modify the environment model with a custom one
        # This part allows us to vary the 5 weather variables
        self.env.unwrapped.model = CustomModelJSON(
            env_name=env_config['env_id'],
            json_file=self.env.unwrapped.building_file,
            weather_files=self.env.unwrapped.weather_files,
            actuators=self.env.unwrapped.actuators,
            variables=self.env.unwrapped.variables,
            meters=self.env.unwrapped.meters,
            max_ep_store=self.env.unwrapped.model.max_ep_store,
            extra_config=self.env.unwrapped.model.config,
        )       

        # Normalize the observation space
        self.env =  NormalizeObservation(self.env)

        # Call the parent constructor
        super(SinergymWrapper, self).__init__(self.env)

    
    def observation(self, observation):
        return observation

    def reset(self, seed=None):
        self.apply_new_weather_var()
        obs, info = self.env.reset(seed=seed)
        return obs

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def apply_new_weather_var(self, file_path='/home/luigi/Documents/SmartEnergyOptimizationRL/data/latest_env_config.json'):
        try:
            with open(file_path, 'r') as f:
                weather_var = json.load(f)
        except FileNotFoundError:
            # Return a default configuration or raise an exception, depending on your preference
            weather_var = {'drybulb': np.array([5.53173187e+00, 0.00000000e+00, 2.55034944e-03])}

        # Apply new weather_variability
        self.env.default_options['weather_variability'] = weather_var