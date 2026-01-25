import argparse
from pathlib import Path
import time
import numpy as np
import yaml

import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.hppo.wrappers import RescaleActionWrapper

from train_scripts.utils import get_wrapper_class

# Register custom envs
import train_scripts.import_envs  # noqa: F401

import importlib

ENV_ID = 'Moving-v0'
ALGO = 'hppo'  # default algorithm (folder/class); can be overridden by CLI


def get_algo_class(algo_name: str):
    """Return the algorithm class from stable_baselines3 for a given name.

    This function will try several strategies:
    1. Look for attributes on the top-level `stable_baselines3` module (e.g. `PPO`, `HPPO`).
    2. If the attribute is a submodule (e.g. `stable_baselines3.hsac`), look for a class inside it
       with a likely name (capitalized or upper-case), or any class that subclasses BaseAlgorithm.
    3. Import `stable_baselines3.<algo_name>` and try the same.
    """
    sb3 = importlib.import_module('stable_baselines3')
    candidates = [algo_name, algo_name.upper(), algo_name.capitalize()]

    # Helper to inspect a module/object for a class
    def find_class_in_obj(obj):
        # try likely class names first
        for cls_name in [algo_name.upper(), algo_name.capitalize()]:
            if hasattr(obj, cls_name):
                cand = getattr(obj, cls_name)
                if isinstance(cand, type):
                    return cand
        # fallback: look for a class that subclasses BaseAlgorithm
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
        except Exception:
            BaseAlgorithm = None
        for v in getattr(obj, '__dict__', {}).values():
            if isinstance(v, type):
                if BaseAlgorithm is None:
                    return v
                try:
                    if issubclass(v, BaseAlgorithm):
                        return v
                except Exception:
                    continue
        return None

    # 1) check top-level attributes
    for c in candidates:
        if hasattr(sb3, c):
            attr = getattr(sb3, c)
            if isinstance(attr, type):
                return attr
            # if it's a module or other object, try to find class inside
            cls = find_class_in_obj(attr)
            if cls is not None:
                return cls

    # 2) try importing stable_baselines3.<algo_name>
    try:
        mod = importlib.import_module(f"stable_baselines3.{algo_name.lower()}")
        cls = find_class_in_obj(mod)
        if cls is not None:
            return cls
    except ModuleNotFoundError:
        pass

    raise ValueError(f"Algorithm class for '{algo_name}' not found in stable_baselines3. Use names like 'ppo','PPO','hppo','HPPO', etc.")

def test_train_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default=str(1), help='index of the model')
    parser.add_argument('--visualize', '-v', action='store_true', help='visualize the simulation')
    parser.add_argument('--lastModel', '-l', action='store_true', help='load the last but not best model')
    parser.add_argument('--algo', '-a', type=str, default=ALGO, help='algorithm name (folder/class), e.g. hppo or HPPO')
    args = parser.parse_args()

    # load the configuration of the environment
    default_path = Path(__file__).parent.parent

    # use the chosen algorithm name for config and model paths
    algo_folder = args.algo.lower()
    config = str(default_path / f"train/hyperparams/{algo_folder}.yml")
    # Load hyperparameters from yaml file
    with open(config) as f:
        hyperparams_dict = yaml.safe_load(f)

    if ENV_ID in list(hyperparams_dict.keys()):
        hyperparams = hyperparams_dict[ENV_ID]
    else:
        raise ValueError(f"Hyperparameters not found for {algo_folder}-{ENV_ID} in {config}")
    
    env_kwargs = hyperparams['eval_env_kwargs'] if 'eval_env_kwargs' in hyperparams else {}
    
    if 'eval_env_name' in list(hyperparams.keys()):
        env_name = hyperparams['eval_env_name']
    else:
        env_name = ENV_ID

    print(f"Using eval env: {env_name}")

    # add render setting
    if args.visualize:
        env_kwargs['render_mode'] = 'human'
    
    spec = gym.spec(env_name)
    # Define make_env here, so it works with subprocesses
    # when the registry was modified with `--gym-packages`
    # See https://github.com/HumanCompatibleAI/imitation/pull/160
    def make_env(**kwargs) -> gym.Env:
        return spec.make(**kwargs)
    
    env_wrapper = get_wrapper_class(hyperparams)

    n_envs = 1
    vec_env = make_vec_env(make_env, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv, wrapper_class=env_wrapper)

    model_index = args.i
    model_cls = get_algo_class(args.algo)
    normalized_env = VecNormalize.load(str(default_path / f'train/models/{algo_folder}/{ENV_ID}_{model_index}/{ENV_ID}/vecnormalize.pkl'), vec_env)
    normalized_env.training = False
    normalized_env.norm_reward = False

    if args.lastModel:
        model_path = default_path / f'train/models/{algo_folder}/{ENV_ID}_{model_index}/rl-replanner-train.zip'
    else:
        model_path = default_path / f'train/models/{algo_folder}/{ENV_ID}_{model_index}/best_model.zip'

    model = model_cls.load(str(model_path), env=normalized_env)

    obs = normalized_env.reset()

    print("Observation space:", normalized_env.observation_space)
    print("Shape of observation space:", normalized_env.observation_space.shape)
    print("Action space:", normalized_env.action_space)
    total_return = 0
    episode_length = 0
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)

        
        
        # print("======================== Step {} ========================".format(step))

        # baseline: always replan trajectory from current position to global goal
        # action = [
        #     {
        #         'id': 1,
        #         "params1": np.array([-2.0, -2.0])
        #     }
        # ]

        # if action[0] == 0:
        # print('Action:', action)

        obs, reward, done, info = normalized_env.step(action)
        info = info[0] if isinstance(info, list) or isinstance(info, tuple) else info
        total_return += reward

        # print('Reward:', reward)
        # print('Observation:', obs)
        # print('Done:', terminated)
        # print('Info:', info)

        if done:
            print("Episode finished after {} timesteps".format(episode_length + 1))
            print("Total return:", total_return)
            print("\n")
            total_return = 0
            episode_length = 0
            step = 0
        
        episode_length += 1
        step += 1
        normalized_env.render()

    vec_env.close()

if __name__ == '__main__':
    test_train_model()
