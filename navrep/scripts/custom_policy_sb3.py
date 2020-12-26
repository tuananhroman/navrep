from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th

from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_RS = 5
_L = 1080


class CustomMlp_arena2d(nn.Module):
    """
    Custom Multilayer Perceptron for policy and value function.
    Architecture was taken as reference from: https://github.com/ignc-research/arena2D/tree/master/arena2d-agents.

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
        super(CustomMlp_arena2d, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Body network
        self.body_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, last_layer_dim_pi),
            nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, last_layer_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        body_x = self.body_net(features)
        return self.policy_net(body_x), self.value_net(body_x)


class CustomMlpPolicy(ActorCriticPolicy):
    """
    Policy using the custom Multilayer Perceptron.
    """

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
        super(CustomMlpPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Enable orthogonal initialization
        self.ortho_init = True

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlp_arena2d(self.features_dim)


class CustomCNN_drl_local_planner(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://github.com/RGring/drl_local_planner_ros_stable_baselines

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomCNN_drl_local_planner, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 256 - _RS),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        observations = observations.transpose(2, 1)

        laser_scan = observations[:, :, :-_RS]
        robot_state = th.squeeze(observations[:, :, -_RS:], 1)

        extracted_features = self.fc_1(self.cnn(laser_scan))
        features = th.cat((extracted_features, robot_state), 1)

        return self.fc_2(features)


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.

:constant policy_drl_local_planner: (dict)
"""
policy_kwargs_drl_local_planner = dict(features_extractor_class=CustomCNN_drl_local_planner,
                                       features_extractor_kwargs=dict(features_dim=128))


class CustomCNN_navrep(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super(CustomCNN_navrep, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim - _RS),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """
        observations = observations.transpose(2, 1)

        laser_scan = observations[:, :, :-_RS]
        robot_state = th.squeeze(observations[:, :, -_RS:], 1)

        extracted_features = self.fc(self.cnn(laser_scan))
        features = th.cat((extracted_features, robot_state), 1)

        return features


"""
Global constant to be passed as an argument to the PPO of Stable-Baselines3 in order to build both the policy
and value network.

:constant policy_kwargs_navrep: (dict)
"""
policy_kwargs_navrep = dict(features_extractor_class=CustomCNN_navrep,
                            features_extractor_kwargs=dict(features_dim=32),
                            net_arch=[dict(vf=[64, 64], pi=[64, 64])], activation_fn=th.nn.ReLU)
