import gymnasium as gym
from sinergym.envs.eplus_env import EplusEnv
from custom_model import CustomModelJSON


class SinergymWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: EplusEnv,
                 env_name: str = 'eplus-env-v1'):
        
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

        # Call the parent constructor
        super(SinergymWrapper, self).__init__(env)


    # TODO: Implement the methods below if necessary
    
    def observation(self, observation):
        return observation

    def reset(self, seed=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
    
    '''
    def separate_resettable_part(self, obs):
        """ Separates the observation into the resettable portion and the original. Make sure this operation is differentiable """
        return obs, obs

    def combine_resettable_part(self, obs, resettable):
        """ Combines an observation that has been split like in separate_resettable_part back together. Make sure this operation is differentiable """
        return resettable

    def sample_obs(self, **kwargs):
        """ Automatically sample an observation to seed state generation """
        return self.observation_space.sample()

    def resettable_bounds(self):
        """ Get bounds for resettable part of observation space """
        return self.observation_space.low, self.observation_space.high
    '''

    def sample_variability(self):
        """ Samples a row from a dataframe containing Ornstein-Uhlenbeck (OU) process parameters for various weather variables. """
        # Sample a row and directly access it
        row = self.OU_params_df.sample(1).iloc[0]
        # Return a dictionary with the sampled values
        return {variable: [row[f"{variable}_{j}"] for j in range(3)] for variable in self.weather_variability}

