"""
registering the customized envs and models for rllib
"""

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from rmp2.envs import *
from rmp2.policies.gaussian_policy import DiagGaussianModel

def register_envs_and_models():
    def franka_env_creator(env_config):
        return FrankaEnv(env_config)
    register_env("franka", franka_env_creator)

    def franka_residual_env_creator(env_config):
        return FrankaResidualEnv(env_config)
    register_env("franka_residual", franka_residual_env_creator)

    def franka_rmp_env_creator(env_config):
        return FrankaResidualRMPEnv(env_config)
    register_env("franka_rmp", franka_rmp_env_creator)

    def three_link_env_creator(env_config):
        return ThreeLinkEnv(env_config)
    register_env("3link", three_link_env_creator)

    def three_link_residual_env_creator(env_config):
        return ThreeLinkResidualEnv(env_config)
    register_env("3link_residual", three_link_residual_env_creator)

    def three_link_rmp_env_creator(env_config):
        return ThreeLinkResidualRMPEnv(env_config)
    register_env("3link_rmp", three_link_rmp_env_creator)

    ModelCatalog.register_custom_model("diag_gaussian_model", DiagGaussianModel)