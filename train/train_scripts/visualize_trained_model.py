"""模型可视化
用法示例：
python train/train_scripts/visualize_trained_model.py --model-dir logs/hsac_dex --env Moving-v0 --output /tmp/vis.gif --max-frames 200
"""
import os
import sys
import argparse
import glob

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import imageio.v2 as imageio
import gymnasium as gym

# Ensure local packages are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SB3_PATH = os.path.join(REPO_ROOT, "algs", "stable-baselines3")
if SB3_PATH not in sys.path:
    sys.path.insert(0, SB3_PATH)
GH_PATH = os.path.join(REPO_ROOT, "envs", "gym-hybrid")
if GH_PATH not in sys.path:
    sys.path.insert(0, GH_PATH)
import gym_hybrid  # registers envs

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HPPO, HSAC


def find_model(path: str):
    if os.path.isdir(path):
        # find newest .zip or .pkl file (prefer zip)
        zips = sorted(glob.glob(os.path.join(path, "**", "*.zip"), recursive=True), key=os.path.getmtime)
        if zips:
            return zips[-1]
        pkls = sorted(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True), key=os.path.getmtime)
        if pkls:
            return pkls[-1]
        return None
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    return None


def resolve_algo_class(data: dict):
    policy_class = data.get("policy_class", None)
    if policy_class is None:
        return None

    module = getattr(policy_class, "__module__", "")
    if ".hppo." in module:
        return HPPO
    if ".hsac." in module:
        return HSAC
    if ".sac." in module:
        return SAC
    if ".td3." in module:
        return TD3
    if ".ddpg." in module:
        return DDPG
    if ".dqn." in module:
        return DQN
    if ".ppo." in module:
        return PPO
    if ".a2c." in module:
        return A2C
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model", default=None, help="Path to model .zip or file")
    p.add_argument("--model-dir", dest="model_dir", default="logs/hsac_dex", help="Directory to search for the latest model")
    p.add_argument("--env", default="Moving-v0")
    p.add_argument("--output", default=None, help="Output GIF path (default: ./vis_{modelname}.gif)")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model_path = args.model or find_model(args.model_dir)
    if model_path is None:
        raise FileNotFoundError("No model found (provide --model or point --model-dir to a folder)")

    print("Using model:", model_path)

    set_random_seed(args.seed)

    from stable_baselines3.common.vec_env import DummyVecEnv
    # Use a DummyVecEnv so the model and policy receive vectorized observations as expected
    env_fn = lambda: gym.make(args.env, render_mode="rgb_array")
    vec_env = DummyVecEnv([env_fn])
    # Inspect saved archive to choose correct class for loading
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _, _ = load_from_zip_file(model_path, load_data=True)
    if data is None:
        raise ValueError("Could not read model archive")

    if "demo_path" in data or "demo_k" in data:
        # HSAC_DEX model
        from stable_baselines3.hsac_dex.hsac_dex import HSAC_DEX
        model = HSAC_DEX.load(model_path, env=vec_env)
    else:
        algo_cls = resolve_algo_class(data)
        if algo_cls is None:
            raise ValueError("Could not resolve algorithm class from saved model; please specify a known algorithm")
        model = algo_cls.load(model_path, env=vec_env)

    frames = []
    obs = vec_env.reset()
    frames.append(vec_env.envs[0].render())

    for _ in range(args.max_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        frames.append(vec_env.envs[0].render())
        if bool(dones[0]):
            break

    out = args.output or os.path.join(".", f"vis_{os.path.splitext(os.path.basename(model_path))[0]}.gif")
    imageio.mimsave(out, frames, fps=20)
    print("Saved GIF to", out)


if __name__ == "__main__":
    main()
