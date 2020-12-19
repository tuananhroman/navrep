from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th

from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

HIDDEN_SHAPE_BODY: int=64
HIDDEN_SHAPE_BODY_OUT: int=64
HIDDEN_SHAPE_POLICY: int=32
HIDDEN_SHAPE_VALUE: int=32

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 32,
            last_layer_dim_vf: int = 32,
    ):
        super(CustomNetwork, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Body network
        self.body_net = nn.Sequential(
            nn.Linear(feature_dim, HIDDEN_SHAPE_BODY),
            nn.ReLU(),
            nn.Linear(HIDDEN_SHAPE_BODY, HIDDEN_SHAPE_BODY_OUT),
            nn.ReLU()
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(HIDDEN_SHAPE_BODY, last_layer_dim_pi),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(HIDDEN_SHAPE_BODY, last_layer_dim_vf),

        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        body_x = self.body_net(features)
        return self.policy_net(body_x), self.value_net(body_x)


class CustomMlp(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            *args,
            **kwargs,
    ):

        super(CustomMlp, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        # Body network
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2),
            nn.ReLU()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))