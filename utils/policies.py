import gym
import torch

from stable_baselines3.common.policies import BasePolicy


class ConstantPolicy(BasePolicy):
    """
    Samples Constant vector
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule = None, 
        use_sde: bool = False,
        response_vector: torch.Tensor = torch.tensor([0., 0.])
    ):
        super(ConstantPolicy, self).__init__(
            observation_space,
            action_space, 
            lr_schedule,
            use_sde
        )
        self.response_vector = response_vector


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass
        :param obs: Observation
        """
        return self.response_vector


    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return constant tensor

        Args:
            observation (torch.Tensor): observation tensor
            deterministic (bool, optional): determinism specifier. Defaults to True.

        Returns:
            torch.Tensor: constant response vector
        """
        action = self.response_vector
        return action