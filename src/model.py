

import numpy as np
import torch
import torch.nn as nn
from typing import Type

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.torch_utils import apply_grad_clipping


class UncertainPPO(PPO):
    @override(PPO)
    def get_default_policy_class(self, config) -> Type[Policy]:
        if config["ppo_framework"] == "torch":
            return UncertainPPOTorchPolicy
        else:
            return NotImplementedError
    

class UncertainPPOTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):

        # Call the parent constructor
        super().__init__(
            observation_space,
            action_space,
            config,
        )
        
        # Set the number of dropout evaluations
        self.num_dropout_evals = config["model"]["num_dropout_evals"]
        # Set the stop gradient flag to False
        self.stop_gradient = False

        print("MODEL INITIALIZED")
    
    def get_value(self, **input_dict):
        """
        Returns the value of the input dictionary
        """
        input_dict = SampleBatch(input_dict)
        input_dict = self._lazy_tensor_dict(input_dict)
        model_out, _ = self.model(input_dict)
        return self.model.value_function()

    def get_action(self, **input_dict):
        """
        Returns the action to take given the input dictionary
        """

        input_dict = SampleBatch(input_dict)
        return self.compute_actions_from_input_dict(input_dict)[0]

    def compute_value_uncertainty(self, obs_tensor: torch.Tensor):
        """
        Computes the uncertainty of the neural network for this observation
        by running inference with different dropout masks and measuring the
        variance of the critic network's output

        Args:
            obs_tensor: torch tensor of observation(s) to compute
                uncertainty for. Make sure it is on the same device
                as the model
        Returns:
            How uncertain the model is about the value for each
                observation
        """

        # Save the original mode of the model
        orig_mode = self.model.training
        # Set the model to training mode
        self.model.train()

        # Create an empty list to store the values
        values = []

        # Loop through the number of dropout evaluations
        for _ in range(self.num_dropout_evals):
            # Get the value of the observation
            vals = self.get_value(obs=obs_tensor, training=True)
            # Append the value to the list
            values.append(vals)
        
        # Stack the values into a tensor
        values = torch.stack(values)
        # Compute uncertainty as the variance of the values
        uncertainty = torch.var(values, dim=0)
        # Set the model back to its original mode
        self.model.train(orig_mode)

        # Return the uncertainty
        return uncertainty

    def compute_reward_uncertainty(self, obs_tensor: torch.Tensor, next_obs_tensor: torch.Tensor):
        """
        Computes the uncertainty of the neural network for the reward
        by running inference with different dropout masks and measuring the
        variance of the critic network's output

        Args:
            obs_tensor: torch tensor of observation(s) to compute
                uncertainty for. Make sure it is on the same device
                as the model
            next_obs_tensor: torch tensor of the next observation(s)
                to compute uncertainty for. Make sure it is on the same
                device as the model
        Returns:
            How uncertain the model is about the reward for each
                observation
        """

        # Save the original mode of the model
        orig_mode = self.model.training
        # Set the model to training mode
        self.model.train()

        # Create an empty list to store the rewards
        rewards = []

        # Loop through the number of dropout evaluations
        for _ in range(self.num_dropout_evals):
            # Get the value of the observation
            curr_val = self.get_value(obs=obs_tensor, training=True)
            # Get the value of the next observation
            next_val = self.get_value(obs=next_obs_tensor, training=True)
            # Compute the reward as the difference between the current value and the next value
            rewards.append(curr_val - self.config["gamma"] * next_val)

        # Concatenate the rewards into a tensor
        rewards = torch.concat(rewards)
        # Since curr_val and next_val are sampled independently
        # the variance of their sum should be the sum of their variances
        # so divide by two so it's still comparable to the planning model's reward uncertainty
        uncertainty = torch.var(rewards, dim=0) / 2
        # Set the model back to its original mode
        self.model.train(orig_mode)

        # Return the uncertainty
        return uncertainty

    def extra_grad_process(self, local_optimizer, loss):
        """
        Applies gradient clipping to the loss.

        Args:
            local_optimizer: A local torch optimizer object.
            loss: The torch loss tensor.
        
        Returns:
            An info dict containing the "grad_norm" key and the resulting clipped
            gradients.
        """
        if self.stop_gradient:
            return self.apply_stop_gradient(local_optimizer, loss)
        else:
            return apply_grad_clipping(self, local_optimizer, loss)
    
    def apply_stop_gradient(self, optimizer, loss):
        """Sets already computed grads inside `optimizer` to 0.

        Args:
            policy: The TorchPolicy, which calculated `loss`.
            optimizer: A local torch optimizer object.
            loss: The torch loss tensor.

        Returns:
            An info dict containing the "grad_norm" key and the resulting clipped
            gradients.
        """

        # Initialize the gradient norm to 0
        grad_gnorm = 0
        # Set the clip value to the value in the config if it exists, otherwise set it to infinity
        if self.config["grad_clip"] is not None:
            clip_value = self.config["grad_clip"]
        else:
            clip_value = np.inf

        # Loop through the parameters in the optimizer
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            # If there are parameters with gradients
            if params:
                # PyTorch clips gradients inplace and returns the norm before clipping
                # We therefore need to compute grad_gnorm further down
                for p in params:
                    p.grad.detach().mul_(0)
                # Clip the gradients
                global_norm = nn.utils.clip_grad_norm_(params, clip_value)

                # If the global norm is a tensor, convert it to a numpy array
                if isinstance(global_norm, torch.Tensor):
                    global_norm = global_norm.cpu().numpy()

                # Add the global norm to the gradient norm
                grad_gnorm += min(global_norm, clip_value)

        # If the gradient norm is greater than 0 return it, otherwise return an empty dictionary
        if grad_gnorm > 0:
            return {"grad_gnorm": grad_gnorm}
        else:
            # No grads available
            return {}
