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

# import the algorithm after registering custom envs
from stable_baselines3 import HPPO

ENV_ID = 'Moving-v0'
ALGO = 'hppo'

def test_train_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default=str(1), help='index of the model')
    parser.add_argument('--visualize', '-v', action='store_true', help='visualize the simulation')
    parser.add_argument('--lastModel', '-l', action='store_true', help='load the last but not best model')
    args = parser.parse_args()

    # load the configuration of the environment
    default_path = Path(__file__).parent.parent
    config = str(default_path / f"train/hyperparams/{ALGO}.yml")
    # Load hyperparameters from yaml file
    with open(config) as f:
        hyperparams_dict = yaml.safe_load(f)

    if ENV_ID in list(hyperparams_dict.keys()):
        hyperparams = hyperparams_dict[ENV_ID]
    else:
        raise ValueError(f"Hyperparameters not found for {ALGO}-{ENV_ID} in {config}")
    
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
    normalized_env = VecNormalize.load(str(default_path / f'train/models/hppo/{ENV_ID}_{model_index}/{ENV_ID}/vecnormalize.pkl'), vec_env)
    normalized_env.training = False
    normalized_env.norm_reward = False

    if args.lastModel:
        model = HPPO.load(str(default_path / f'train/models/hppo/{ENV_ID}_{model_index}/rl-replanner-train.zip'), env=normalized_env)
    else:
        model = HPPO.load(str(default_path / f'train/models/hppo/{ENV_ID}_{model_index}/best_model.zip'), env=normalized_env)

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
