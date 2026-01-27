import argparse
import os
import sys
import time
import glob

import numpy as np
import gymnasium as gym

# Ensure local stable-baselines3 package is importable when running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SB3_PATH = os.path.join(REPO_ROOT, "algs", "stable-baselines3")
if SB3_PATH not in sys.path:
    sys.path.insert(0, SB3_PATH)

GH_PATH = os.path.join(REPO_ROOT, "envs", "gym-hybrid")
if GH_PATH not in sys.path:
    sys.path.insert(0, GH_PATH)
import gym_hybrid  # registers Moving-v0 and Sliding-v0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-path", required=True, help="Path to a single expert demo .npz file or a directory with .npz files")
    p.add_argument("--env", default="Moving-v0", help="Gym env id")
    p.add_argument("--n-timesteps", default=100000, type=float, help="Number of timesteps to train for")
    p.add_argument("--device", default="cpu", help="PyTorch device to use (cpu or cuda)")
    p.add_argument("--save-dir", default="logs/hsac_dex", help="Directory where model will be saved")
    p.add_argument("--demo-batch-size", default=256, type=int)
    p.add_argument("--demo-aux-weight", default=1.0, type=float)
    p.add_argument("--demo-k", default=5, type=int)
    p.add_argument("--learning-rate", default=3e-4, type=float)
    p.add_argument("--buffer-size", default=100000, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--train-freq", default=1, type=int)
    p.add_argument("--gradient-steps", default=1, type=int)
    p.add_argument("--learning-starts", default=1000, type=int)
    return p.parse_args()


def verify_demo(demo_path: str) -> None:
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if os.path.isdir(demo_path):
        files = sorted(glob.glob(os.path.join(demo_path, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz demo files found in directory: {demo_path}")
        print(f"Inspecting demo directory: {demo_path} (found {len(files)} files)")
        for f in files:
            with np.load(f) as data:
                print(f" - {os.path.basename(f)}: {list(data.keys())}")
    else:
        with np.load(demo_path) as data:
            print(f"Inspecting demo file: {os.path.basename(demo_path)}: {list(data.keys())}")


def main(args=None):
    if args is None:
        args = parse_args()
    demo_path = args.demo_path
    verify_demo(demo_path)

    # Normalize device aliases and check torch availability
    device = args.device
    if device == "gpu":
        device = "cuda"

    import torch as th
    if device == "cuda" and not th.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU")
        device = "cpu"

    args.device = device

    # Import HSAC_DEX lazily (after verifying demo readability)
    # direct import; let ImportError surface if package missing
    from stable_baselines3.hsac_dex.hsac_dex import HSAC_DEX

    print("Creating environment:", args.env)
    env = gym.make(args.env)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = int(time.time())
    save_name = os.path.join(args.save_dir, f"hsac_dex_{args.env}_{timestamp}")

    print("Instantiating HSAC_DEX with demo_path", demo_path)
    # Provide a default policy name (MlpPolicy) to satisfy HSAC constructor
    model = HSAC_DEX(
        "MlpPolicy",
        env=env,
        demo_path=demo_path,
        demo_batch_size=args.demo_batch_size,
        demo_aux_weight=args.demo_aux_weight,
        demo_k=args.demo_k,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=1,
    )

    print(f"Starting training for {int(args.n_timesteps)} timesteps...")
    model.learn(int(args.n_timesteps))

    print("Saving model to", save_name)
    model.save(save_name)

    if hasattr(model, "save_replay_buffer"):
        rb_path = save_name + "_replay_buffer.pkl"
        print("Saving replay buffer to", rb_path)
        model.save_replay_buffer(rb_path)
    # Close env
    model.env.close()

    print("Done.")


if __name__ == "__main__":
    main()
