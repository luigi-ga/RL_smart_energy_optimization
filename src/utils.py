# Standard library imports
from bisect import insort
from copy import deepcopy
from enum import Enum
import pickle
import random

# Third-party library imports
import cooper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Dict, Discrete, Space

# ray imports
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID

# Local application/library specific imports
from src.model import UncertainPPO

# Typing imports (deduplicated)
from typing import Dict, Optional, Union



# ACTIVE_STATE_VISITATION_KEY = "active_state_visitation"
UNCERTAINTY_LOSS_KEY = "uncertainty_loss"
CL_ENV_KEYS = ["cold_Texas", "dry_Cali", "hot_new_york", "snowy_Cali_winter"]
DEFAULT_REW_MAP = {
    b'E': -0.2,
    b'S': -0.2,
    b'W': -5.0,
    b'G': 10.0,
    b'U': -0.2,
    b'R': -0.2,
    b"L": -0.2,
    b"D": -0.2,
    b"B": -10.0
}

RESPAWNABLE_TOKENS = [".", "P"]
class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"
    DM_MAZE = "dm"
    SINERGYM = "sg"

class SG_WEATHER_TYPES(Enum):
    HOT = "hot"
    COOL = "cool"
    MIXED = "mixed"

class EnvBufferEntry:
    def __init__(self, value_loss, env_params, last_seen=1) -> None:
        self.value_error = np.abs(value_loss)
        self.env_params  = env_params
        self.last_seen = last_seen

    def __repr__(self) -> str:
        return f"{self.value_error}, {self.env_params}, {self.last_seen}"

'''
def states_to_np(state, inplace=True):
    if not inplace:
        state = deepcopy(state)
    if isinstance(state, dict):
        for k, v in state.items():
            state[k] = v.detach().squeeze().cpu().numpy()
        return state
    elif isinstance(state, np.ndarray):
        return state
    else:
        return state.detach().squeeze().cpu().numpy()
'''
        
class LinearDecayScheduler():
    """Parameter scheduler that linearly increases p to 1 and stays there."""
    def __init__(self, envs_to_1=200) -> None:
        self.envs_to_1 = envs_to_1
        self.current_value = 0

    def step(self, env_buffer):
        num_seen = len(env_buffer)
        self.current_value = num_seen / self.envs_to_1
        return self.current_value

class ActiveRLCallback(DefaultCallbacks):
    """
    A custom callback that derives from ``DefaultCallbacks``. Not yet vectorized.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, 
                 num_descent_steps: int=10, 
                 batch_size: int=64, 
                 no_coop: bool=False, 
                 planning_model=None, 
                 config={}, 
                 run_active_rl=0, 
                 planning_uncertainty_weight=1, 
                 device="cpu", 
                 args={}, 
                 uniform_reset = 0):
        super().__init__()
        self.run_active_rl = run_active_rl
        self.num_descent_steps = num_descent_steps
        self.batch_size = batch_size
        self.no_coop = no_coop
        self.planning_model = planning_model
        self.config = config
        self.is_evaluating = False
        self.planning_uncertainty_weight = planning_uncertainty_weight
        self.use_gpu = args.num_gpus > 0
        self.args = args
        self.uniform_reset = uniform_reset
        self.full_eval_mode = False
        self.activerl_lr = args.activerl_lr
        self.activerl_reg_coeff = args.activerl_reg_coeff
        self.eval_worker_ids = []
        if self.planning_model is not None:
            device = "cuda:0" if self.use_gpu else "cpu"
            self.reward_model = RewardPredictor(self.planning_model.obs_size, self.config["model"]["fcnet_hiddens"][0], False, device=device)
            self.reward_optim = torch.optim.Adam(self.reward_model.parameters())
        else:
            self.reward_model = None
            self.reward_optim = None

        self.plr_d = args.plr_d
        self.plr_beta = args.plr_beta if hasattr(args, "plr_beta") else 0.1
        self.plr_rho = args.plr_rho  if hasattr(args, "plr_rho") else 0.1
        self.plr_envs_to_1 = args.plr_envs_to_1 if hasattr(args, "plr_envs_to_1") else 1e6
        self.plr_robust = args.plr_robust if hasattr(args, "plr_robust") else False
        self.naive_grounding = args.naive_grounding if hasattr(args, "naive_grounding") else False
        print("THIS IS NAIVE GROUNDING", self.naive_grounding)
        self.env_buffer = []
        self.plr_scheduler = LinearDecayScheduler(self.plr_envs_to_1)
        self.last_reset_state = None
        self.next_sampling_used = None
        self.next_initial_state = None
        self.env_repeat = args.env_repeat
        self.num_train_steps = 0 
        self.start = args.start

    def on_evaluate_start(self, *, algorithm: UncertainPPO, **kwargs)-> None:
        """
        This method gets called at the beginning of Algorithm.evaluate().
        """

        def activate_eval_metrics(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = True
            return worker.worker_index
        
        def set_eval_worker_ids(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.eval_worker_ids = self.eval_worker_ids

        self.eval_worker_ids = sorted(algorithm.evaluation_workers.foreach_worker(activate_eval_metrics))
        algorithm.evaluation_workers.foreach_worker(set_eval_worker_ids)


    def on_evaluate_end(self, *, algorithm: UncertainPPO, evaluation_metrics: dict, **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        #self.is_evaluating = False
        def access_eval_metrics(worker):
            ###
            if hasattr(worker, "callbacks"):
                worker.callbacks.is_evaluating = False
            else:
                return []
        return algorithm.evaluation_workers.foreach_worker(access_eval_metrics)
        
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[0]
        # Get the single "default policy"
        policy = next(policies.values())
        run_active_rl = np.random.random() < self.run_active_rl
        if not self.is_evaluating and (run_active_rl or self.uniform_reset) and self.next_initial_state is not None:
            self.reset_env(policy, env, episode)

    def _get_next_initial_state(self, policy, env, env_buffer=[]):
        plr_d = self.plr_scheduler.step(env_buffer)
        print("THIS IS HOW BIG THE ENV BUFFER ISSSSSSSSSSSSSSSSSSSSSSSSSSSSSS", len(env_buffer))

        # Repeat the environment self.env_repeat times
        if (self.num_train_steps % self.env_repeat) != 0 and self.next_initial_state is not None:
            print(f"REPEATING ENVIRONMENT ON STEP {self.num_train_steps}")
            return self.next_initial_state, self.next_sampling_used
        
        new_states, uncertainties, sampling_used = generate_states(
            policy, 
            env=env, 
            obs_space=env.observation_space, 
            curr_iter=self.num_train_steps,
            num_descent_steps=self.num_descent_steps, 
            batch_size=self.batch_size, 
            no_coop=self.no_coop, 
            planning_model=self.planning_model, 
            reward_model=self.reward_model, 
            planning_uncertainty_weight=self.planning_uncertainty_weight, 
            uniform_reset=self.uniform_reset,
            lr=self.activerl_lr,
            plr_d=plr_d,
            plr_beta=self.plr_beta,
            plr_rho=self.plr_rho,
            env_buffer=env_buffer,
            reg_coeff = self.activerl_reg_coeff, 
            naive_grounding=self.naive_grounding)
        new_states = self.states_to_np(new_states)
        # episode.custom_metrics[UNCERTAINTY_LOSS_KEY] = uncertainties[-1] # TODO: PUT THIS BACK IN SOMEWHERE
        self.next_sampling_used = sampling_used
        return new_states, sampling_used

    @staticmethod
    def states_to_np(state, inplace=True):
        if not inplace:
            state = deepcopy(state)
        if isinstance(state, dict):
            for k, v in state.items():
                state[k] = v.detach().squeeze().cpu().numpy()
            return state
        elif isinstance(state, np.ndarray):
            return state
        else:
            return state.detach().squeeze().cpu().numpy()

    def reset_env(self, policy, env, episode):
        
        env.reset(initial_state=self.next_initial_state)
        
        return self.next_initial_state

    def on_learn_on_batch(self, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs):
        """
        Runs each time the agent is updated on a batch of data (Note this also applies to PPO's minibatches)

        Args:
            worker: Reference to the current rollout worker.
            policies : Mapping of policy id to policy objects. In single agent mode
                there will only be a single “default_policy”.
            train_batch: Batch of training data
            result : Dictionary to add custom metrics into
            kwargs : Forward compatibility placeholder.
        """
        if self.reward_model is not None:
            obs = torch.tensor(train_batch[SampleBatch.OBS], device=self.reward_model.device)
            rew = torch.tensor(train_batch[SampleBatch.REWARDS], device=self.reward_model.device)
            self.reward_optim.zero_grad()
            rew_hat = self.reward_model(obs).squeeze()
            loss = F.mse_loss(rew, rew_hat)
            result["reward_predictor_loss"] = loss.detach().cpu().numpy()
            loss.backward()
            self.reward_optim.step()

    def on_train_result(
        self,
        *,
        algorithm: str = "Algorithm",
        result: dict = {},
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        self.num_train_steps += 1
        if self.num_train_steps <= self.start:
            return


        def set_next_initial_state(worker: RolloutWorker):
            if hasattr(worker, 'callbacks'):
                worker.callbacks.last_reset_state = worker.callbacks.next_initial_state
                worker.callbacks.next_initial_state = self.next_initial_state
                worker.callbacks.next_sampling_used = self.next_sampling_used

        

        def get_candidate_initial_states(worker: RolloutWorker):
            if hasattr(worker, 'callbacks') and worker.env is not None:
                worker.callbacks.num_train_steps = self.num_train_steps
                return worker.callbacks._get_next_initial_state(worker.get_policy(), worker.env, self.env_buffer)
            return None

        self.next_initial_state, self.next_sampling_used = next(filter(lambda x: x!=None, algorithm.workers.foreach_worker(get_candidate_initial_states)))

        algorithm.workers.foreach_worker(set_next_initial_state)
        
        

        if self.plr_d > 0 and self.last_reset_state is not None:
            stop_gradient = False
            # Update staleness parameters in the env buffer for the next training iteration
            if self.next_sampling_used == "PLR":
                self.update_env_last_seen(self.next_initial_state, self.num_train_steps)
                print(f"NOT Setting learning rate to 0 on step {self.num_train_steps}")
            else:
                # Insert the env that was just seen during this iteration into the env_buffer
                vf_loss = result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]
                entry = EnvBufferEntry(np.abs(vf_loss), self.last_reset_state, len(self.env_buffer))
                insort(self.env_buffer, entry, key=lambda x: -x.value_error)
                if self.plr_robust:
                    # Set the algorithm's learning rate to 0 if next_sampling_used != "PLR"
                    stop_gradient = True
                    print(f"Setting learning rate to 0 on step {self.num_train_steps}")
            def set_stop_gradient(policy, policy_id):
                policy.stop_gradient = stop_gradient
            algorithm.workers.foreach_policy(set_stop_gradient)

        self.last_reset_state = self.next_initial_state
        

    def update_env_last_seen(self, env_params, i):
        """
        Searches through self.env_buffer for an env with the same parameters as env_entry and
        updates the entry's last-seen variable to i.
        """
        for entry in self.env_buffer:
            if np.all(entry.env_params == env_params):
                entry.last_seen = i

    def full_eval(self, algorithm):
        """
            Sets callback into full evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = True
        def set_full_eval(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.full_eval_mode = True
        algorithm.evaluation_workers.foreach_worker(set_full_eval)

    def limited_eval(self, algorithm):
        """
            Sets callback into limited evaluation mode. Similar to pytorch\'s eval function,
            this does not *actually* run any evaluations
        """
        self.full_eval_mode = False
        def set_limited_eval(worker):
            if hasattr(worker, "callbacks"):
                worker.callbacks.full_eval_mode = False
        algorithm.evaluation_workers.foreach_worker(set_limited_eval)
     
class SinergymCallback(ActiveRLCallback):
    def __init__(self, 
                 num_descent_steps: int=10, 
                 batch_size: int=64, 
                 no_coop: bool=False, 
                 planning_model=None, 
                 config={}, 
                 run_active_rl=0, 
                 planning_uncertainty_weight=1, 
                 device="cpu", 
                 args={}, 
                 uniform_reset=0):
        super().__init__(num_descent_steps, batch_size, no_coop, 
                         planning_model, config, run_active_rl, 
                         planning_uncertainty_weight, device, args, 
                         uniform_reset)
        self.num_envs = config["num_envs_per_worker"]
        self.env_to_scenario_index = {k: -1 for k in range(self.num_envs)}
        self.sample_environments = config["env_config"].get("sample_environments", False)
        self.scenario_index = 0

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        episode.user_data["power"] = []
        episode.user_data["term_comfort"] = []
        episode.user_data["term_energy"] = []
        episode.user_data["num_comfort_violations"] = 0
        episode.user_data["out_temperature"] = []
        episode.user_data[f"reward"]= []

        env = base_env.get_sub_environments()[env_index]
        # Get the single "default policy"
        policy = next(policies.values())
        train_reset = np.random.random() < max(self.run_active_rl, self.uniform_reset)
        if not self.is_evaluating and train_reset:
            self.reset_env(policy, env, episode)
        elif self.is_evaluating:
            is_default_env_worker = (worker.worker_index == self.eval_worker_ids[0]) and env_index == 0
            if self.sample_environments and not is_default_env_worker:
                # Set scenario_index to -2 to sample weather variability.
                # We also want to make sure the default environment is represented,
                # so let one environment reset with the default variability.
                scenario_index = -2
                # self.scenario_index = (self.scenario_index + 1) % (self.num_envs * len(self.eval_worker_ids))
            else:
                scenario_index = self.scenario_index
                self.scenario_index = (self.scenario_index + 1) % len(env.weather_variability)
            print("WHAT IS MY SCENARIO INDEX???", scenario_index)
            env.reset(scenario_index)
            self.env_to_scenario_index[env_index] = scenario_index
            

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[env_index]
        obs = episode.last_observation_for()
        info = episode.last_info_for()
        episode.user_data["power"].append(info["total_power"])
        episode.user_data["term_comfort"].append(info["comfort_penalty"])
        episode.user_data["term_energy"].append(info["total_power_no_units"])
        episode.user_data["out_temperature"].append(info["out_temperature"])
        episode.user_data[f"reward"].append(episode.last_reward_for())
        if info["comfort_penalty"] != 0:
            episode.user_data["num_comfort_violations"] += 1

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs)-> None:
        """
        Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env : BaseEnv running the episode. The underlying sub environment
                objects can be retrieved by calling base_env.get_sub_environments().
            policies : Mapping of policy id to policy objects. In single agent mode
                there will only be a single “default_policy”.
            episode : Episode object which contains episode state. You can use the
                episode.user_data dict to store temporary data, and episode.custom_metrics
                to store custom metrics for the episode. In case of environment failures,
                episode may also be an Exception that gets thrown from the environment
                before the episode finishes. Users of this callback may then handle
                these error cases properly with their custom logics.
            kwargs : Forward compatibility placeholder.
        """
        to_log = {}
        to_log["cum_power"] = np.sum(episode.user_data["power"])
        to_log["mean_power"] = np.mean(episode.user_data["power"])
        to_log["cum_comfort_penalty"] = np.sum(episode.user_data["term_comfort"])
        to_log["mean_comfort_penalty"] = np.mean(episode.user_data["term_comfort"])
        to_log["cum_power_penalty"] = np.sum(episode.user_data["term_energy"])
        to_log["mean_power_penalty"] = np.mean(episode.user_data["term_energy"])
        to_log["num_comfort_violations"] = episode.user_data["num_comfort_violations"]
        to_log["out_temperature_mean"] = np.mean(episode.user_data["out_temperature"])
        to_log["out_temperature_std"] = np.std(episode.user_data["out_temperature"])
        to_log["reward_mean"] = np.mean(episode.user_data["reward"])
        to_log["reward_sum"] = np.sum(episode.user_data["reward"])
        episode.hist_data["out_temperature"] = episode.user_data["out_temperature"][::6000]
        
        try:
            to_log['comfort_violation_time(%)'] = episode.user_data["num_comfort_violations"] / \
                episode.length * 100
        except ZeroDivisionError:
            to_log['comfort_violation_time(%)'] = np.nan

        # Log both scenario specific and aggregated logs
        episode.custom_metrics.update(to_log)
        scenario_index = self.env_to_scenario_index[env_index]
        env_specific_log = {f"env_{scenario_index}_{key}": val for key, val in to_log.items()}
        episode.custom_metrics.update(env_specific_log)


#### EPW DATA #####################################################################################

class EPW_Data:
    def __init__(self, epw_df=None, transformed_df=None, pca=None, OU_mean=None, OU_std=None, 
                 OU_min=None, OU_max=None, weather_min=None, weather_max=None, base_OU=None) -> None:
        self.epw_df = epw_df
        self.transformed_df = transformed_df
        self.pca = pca
        self.OU_mean = OU_mean
        self.OU_std = OU_std
        self.OU_min = OU_min
        self.OU_max = OU_max
        self.base_OU = base_OU
        self.weather_min = weather_min
        self.weather_max = weather_max
    
    def read_OU_param(self, df, name):
        """
        Helper function to read the 3 OU parameters corresponding to a certain variable name in
        epw_df, transformed_df, OU_mean, or OU_std. 
        """
        return np.array([df[f"{name}_{i}"] for i in range(3)])
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(vars(self), f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            
            variables =  pickle.load(f)
        return EPW_Data(**variables)
    

### STATE GENERATION    #####################################################################################

class BoundedUncertaintyMaximization(cooper.ConstrainedMinimizationProblem):
    """Constrained minimization problem that maximizes the uncertainty of the agent's value function."""
    def __init__(self, obs, env, lower_bounds, upper_bounds, lower_bounded_idxs, upper_bounded_idxs, agent, planning_model=None, reward_model=None, planning_uncertainty_weight=1, reg_coeff = 0.01):
        self.env = env
        self.obs = obs
        self.original_obs = self.obs.detach()
        self.original_resettable, _ = env.separate_resettable_part(self.original_obs)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.lower_bounded_idxs = lower_bounded_idxs
        self.upper_bounded_idxs = upper_bounded_idxs
        self.agent = agent
        self.planning_model = planning_model
        self.reward_model = reward_model
        self.planning_uncertainty_weight = planning_uncertainty_weight
        self.reg_coeff = reg_coeff
        super().__init__(is_constrained=True)

    def closure(self, resettable):
        obs = self.env.combine_resettable_part(self.obs, resettable)
        
        if self.planning_model is None:
            # Negative sign added since we want to *maximize* the uncertainty
            loss = - self.agent.compute_value_uncertainty(obs).sum()
        else:
            action = self.agent.get_action(obs=obs)
            planning_uncertainty, next_obs = self.planning_model.compute_reward_uncertainty(obs, action, return_avg_state=True)
            agent_uncertainty = self.reward_model.compute_uncertainty(obs)
            denom = 1 + self.planning_uncertainty_weight
            loss = (- agent_uncertainty + self.planning_uncertainty_weight * planning_uncertainty) / denom
            #print("IS THIS LOSS? ", loss, loss.shape)

        # Entries of p >= 0 (equiv. -p <= 0)
        #resettable, obs = self.env.separate_resettable_part(obs)
        loss += self.reg_coeff * torch.norm(resettable - self.original_resettable.detach())
        ineq_defect = torch.cat([resettable[self.lower_bounded_idxs] - self.lower_bounds, self.upper_bounds - resettable[self.upper_bounded_idxs]])
        return cooper.CMPState(loss=loss, ineq_defect=ineq_defect)

def get_space_bounds(obs_space: Space):
    """Returns the lower and upper bounds of the observation space."""
    if isinstance(obs_space, Box):
        return obs_space.low, obs_space.high
    elif isinstance(obs_space, Discrete):
        return np.atleast_1d(obs_space.start), np.atleast_1d(obs_space.start + obs_space.n)
    else:
        raise NotImplementedError

def sample_obs(env, batch_size: int, device, random: bool=False, naive_grounding=False):
    """Samples an observation from the environment to seed state generation. If random is True, 
        samples random observations from the observation space. Otherwise, uses environment's sample_obs method.
        Specifying that you are using naive grounding will override the random flag."""
    obs_space = env.observation_space
    if isinstance(obs_space, Dict):
        obs = {}
        with torch.no_grad():
            for _ in range(batch_size):
                if random and not naive_grounding:
                    sampled_obs = env.observation_space.sample()
                else:
                    sampled_obs = env.sample_obs(naive_grounding=naive_grounding)
                if isinstance(sampled_obs, dict):
                    for key, val in sampled_obs.items():
                        if key not in obs:
                            obs[key] = []
                        obs[key].append(torch.tensor(val, device = device, dtype=torch.float32, requires_grad=False))
        obs = {k: torch.stack(v) for k, v in obs.items()}
    else:
        obs = []
        with torch.no_grad():
            for i in range(batch_size):
                if random and not naive_grounding:
                    sampled_obs = env.observation_space.sample()
                else:
                    sampled_obs = env.sample_obs(naive_grounding=naive_grounding)
                obs.append(torch.tensor(sampled_obs, device = device, dtype=torch.float32, requires_grad=False))

        obs = torch.stack(obs)
    resettable_part, obs = env.separate_resettable_part(obs)
    
    resettable_part = torch.nn.Parameter(resettable_part.detach(), requires_grad=True) # Make a leaf tensor that is an optimizable Parameter
    obs = env.combine_resettable_part(obs, resettable_part)
    return obs, resettable_part

def random_state(lower_bounds: np.ndarray, upper_bounds: np.ndarray):
    """Returns a random state that was uniform randomly sampled between lower_bounds and upper_bounds"""
    ret = np.random.random(lower_bounds.shape)
    ret -= lower_bounds
    ret *= (upper_bounds - lower_bounds)
    return ret

def plr_sample_state(env_buffer, beta, curr_iter, rho=0.1):
    """Samples a state from the environment buffer using the PLR algorithm."""
    assert len(env_buffer) > 0
    assert beta != 0

    ranks = np.arange(1, len(env_buffer) + 1)
    score_prioritized_p = (1 / ranks) ** (1 / beta)
    score_prioritized_p /= np.sum(score_prioritized_p)

    staleness = np.array([curr_iter - env_entry.last_seen for env_entry in env_buffer])
    staleness_p = staleness / max(1, np.sum(staleness))
    
    p = (1 - rho) * score_prioritized_p + rho * staleness_p
    sampled_env_entry = random.choices(env_buffer, k=1, weights=p)[0]
    return sampled_env_entry.env_params

def generate_states(agent, env, obs_space: Space, curr_iter: int,
                    num_descent_steps: int = 10, batch_size: int = 1, no_coop=False, 
                    planning_model=None, reward_model=None, planning_uncertainty_weight=1, 
                    uniform_reset=False, lr=0.1, plr_d=0.0, plr_beta=0.1, env_buffer=[], reg_coeff=0.01, plr_rho=0.1, naive_grounding=False):
    """
        Generates states by doing gradient descent to increase an agent's uncertainty
        on states starting from random noise

        :param agent: the agent 
        :param env: an environment that implements the separate_resettable_part and combine_resettable_part methods
        :param obs_space: the observation space
        :param num_descent_steps: the number of gradient descent steps to do
        :param batch_size: the number of observations to concurrently process (CURRENTLY DOESN'T DO ANYTHING, JUST SET IT TO 1)
        :param no_coop: whether or not to use the constrained optimization solver coop to make sure we don't go out of bounds. WILL LIKELY FAIL IF NOT SET TO TRUE
        :param planning_model: the planning model that was trained offline
        :param reward_model: the reward model you are training online
        :param projection_fn: a function to project a continuous, unconstrained observation vector
                                to the actual observation space (e.g. if the observation space is actually
                                discrete then you can round the features in the observation vector)
        :param planning_uncertainty_weight: relative weight to give to the planning uncertainty compared to agent uncertainty
        :param uniform_reset: whether to just sample uniform random from the resettable_bounds space
        :param lr: learning rate for both primal and dual optimizers
        :param env_buffer: A list of tuples (|value loss|, env_parameters) sorted in descending order by |value loss|
        :param reg_coeff: regularization coefficient for the dual optimizer
        :param plr_d: Whether or not to use PLR
        :param plr_beta: PLR beta parameter
        :param plr_rho: PLR rho parameter
        :return: a batch of observations
    """
#     #TODO: make everything work with batches
    lower_bounds, upper_bounds = env.resettable_bounds()#get_space_bounds(obs_space)
    lower_bounded_idxs = np.logical_not(np.isinf(lower_bounds))
    upper_bounded_idxs = np.logical_not(np.isinf(upper_bounds))

    use_plr = np.random.random() < plr_d
    if use_plr and len(env_buffer) > 0:
        print("USING PRIORITIZED LEVEL REPLAY")
        return plr_sample_state(env_buffer, plr_beta, curr_iter, rho=plr_rho), [0], "PLR"
    
    obs, resettable = sample_obs(env, batch_size, agent.device, random=uniform_reset, naive_grounding=naive_grounding)

    if uniform_reset:
        print("USING UNIFORM RESET")
        return obs, [0], "UNIFORM"
    
    if not no_coop:
        cmp = BoundedUncertaintyMaximization(
                                                obs,
                                                env,
                                                torch.tensor(lower_bounds[lower_bounded_idxs], device=agent.device), 
                                                torch.tensor(upper_bounds[upper_bounded_idxs], device=agent.device), 
                                                torch.tensor(lower_bounded_idxs[None, :], device=agent.device), 
                                                torch.tensor(upper_bounded_idxs[None, :], device=agent.device), 
                                                agent,
                                                planning_model,
                                                reward_model,
                                                planning_uncertainty_weight,
                                                reg_coeff
                                                )
        formulation = cooper.LagrangianFormulation(cmp)

        primal_optimizer = cooper.optim.ExtraAdam([resettable], lr=lr)

        # Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
        # yet. Cooper takes care of this, once it has initialized the formulation state.
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=lr)

        # Wrap the formulation and both optimizers inside a ConstrainedOptimizer
        optimizer = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)
    else:
        optimizer = optim.Adam([resettable], lr=lr)
    uncertainties = []
    
    for _ in range(num_descent_steps):
        optimizer.zero_grad()
        agent.model.zero_grad()
        if not no_coop:
            lagrangian = formulation.composite_objective(cmp.closure, resettable)
            formulation.custom_backward(lagrangian)
            optimizer.step(cmp.closure, resettable)
            uncertainties.append(cmp.state.loss.detach().cpu().numpy())
        else:
            obs = env.combine_resettable_part(obs, resettable)
            uncertainty = agent.compute_value_uncertainty(obs)
            loss = - uncertainty.sum()
            loss.backward()
            optimizer.step()
            uncertainties.append(uncertainty.detach().cpu().numpy())
    return obs, uncertainties, "ACTIVERL"


### REWARD PREDICTOR    #####################################################################################
'''
def get_unit(in_size, out_size, batch_norm=True, activation=nn.ReLU):
    return nn.Sequential(
        nn.Linear(in_size, out_size), 
        activation(),
        nn.BatchNorm1d(out_size) if batch_norm else nn.Identity(),
        nn.Dropout(),
        )
'''
class RewardPredictor(nn.Module):
    """
    Neural network to predict reward of next state given the current observation. Currently implemented as 
    a 3 layer neural network (2 hidden layers) with customizable hidden unit size and activation.

    :param in_size: size of input
    :param hidden_size: size of hidden units
    :param batch_norm: whether or not to use batch norm in each hidden layer
    :param device: what device to initialize this to
    :param activation: what activation function to use
    """
    def __init__(self, in_size, hidden_size, batch_norm: bool=False, activation=nn.Tanh, device="cpu") -> None:
        super().__init__()
        self.X_mean = nn.Parameter(torch.zeros(in_size), requires_grad=False)
        self.X_std = nn.Parameter(torch.ones(in_size), requires_grad=False)
        self.batch_norm = batch_norm
        self.y_mean = nn.Parameter(torch.zeros([1]), requires_grad=False)
        self.y_std = nn.Parameter(torch.ones([1]), requires_grad=False)
        self.momentum = 0.9
        self.layers = nn.ModuleList([
            self.get_unit(in_size, hidden_size, batch_norm, activation=activation),
            self.get_unit(hidden_size, hidden_size, batch_norm, activation=activation),
            nn.Linear(hidden_size, 1)
        ])
        self.device = device
        self.to(device)
    
    @staticmethod
    def get_unit(in_size, out_size, batch_norm=True, activation=nn.ReLU):
        return nn.Sequential(
            nn.Linear(in_size, out_size), 
            activation(),
            nn.BatchNorm1d(out_size) if batch_norm else nn.Identity(),
            nn.Dropout(),
            )

    def preprocess(self, x):
        ret = (x - self.X_mean.to(self.device)) / self.X_std.to(self.device)
        # Do not update on single samples
        if self.training and len(x) > 1:
            self.X_mean.data = self.momentum * self.X_mean + (1 - self.momentum) * torch.mean(x)
            self.X_std.data = self.momentum * self.X_std + (1 - self.momentum) * torch.std(x)
        return ret

    def postprocess(self, y):
        ret = y * self.y_std.to(self.device) + self.y_mean.to(self.device)
        # Do not update on single samples
        if self.training and len(y) > 1:
            self.y_mean.data = self.momentum * self.y_mean + (1 - self.momentum) * torch.mean(y)
            self.y_std.data = self.momentum * self.y_std + (1 - self.momentum) * torch.std(y)
        return ret
    
    def forward(self, x):
        x = self.preprocess(x)
        for i, layer in enumerate(self.layers):
            base = 0
            # Add residual connection if this is not
            # the first or last layer
            if i != 0 and i != len(self.layers) - 1:
                base = x
            x = layer(x) + base
        return self.postprocess(x)

    def eval_batchnorm(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.BatchNorm1d):
                        sublayer.eval()

    def compute_uncertainty(self, in_tensor, num_dropout_evals=10):
        orig_mode = self.training
        self.train()
        self.eval_batchnorm()
        rewards = []
        for _ in range(num_dropout_evals):
            rew = self.forward(in_tensor)
            rewards.append(rew)
        rewards = torch.stack(rewards)
        uncertainty = torch.mean(torch.var(rewards, axis=0))
        self.train(orig_mode)
        return uncertainty