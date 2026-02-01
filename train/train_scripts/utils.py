import argparse
import glob
import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import stable_baselines3 as sb3  # noqa: F401
import torch as th  # noqa: F401
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HPPO, HSAC
from stable_baselines3.hsac_dex.hsac_dex import HSAC_DEX
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

# For custom activation fn
from torch import nn as nn

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # Hybrid PPO
    "hppo": HPPO,
    # Hybrid SAC
    "hsac": HSAC,
    # DEX-style HSAC with demo guidance
    "hsac_dex": HSAC_DEX,
}


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, spaces.Dict)
    return gym.wrappers.FlattenObservation(env)


def get_wrapper_class(hyperparams: Dict[str, Any], key: str = "env_wrapper") -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    Works also for VecEnvWrapper with the key "vec_env_wrapper".

    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - rl_zoo3.wrappers.PlotActionWrapper
        - rl_zoo3.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if key in hyperparams.keys():
        wrapper_name = hyperparams.get(key)

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = next(iter(wrapper_dict.keys()))
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_class_by_name(name: str) -> Type:
    """
    Imports and returns a class given the name, e.g. passing
    'stable_baselines3.common.callbacks.CheckpointCallback' returns the
    CheckpointCallback class.

    :param name:
    :return:
    """

    def get_module_name(name: str) -> str:
        return ".".join(name.split(".")[:-1])

    def get_class_name(name: str) -> str:
        return name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - rl_zoo3.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    callbacks: List[BaseCallback] = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = next(iter(callback_dict.keys()))
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}

            callback_class = get_class_by_name(callback_name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def get_latest_run_id(log_path: str, env_name: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_name:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, env_name + "_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(env_name) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)
