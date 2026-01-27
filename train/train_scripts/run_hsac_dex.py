"""Run a simple HSAC_DEX training using local demo .npz files.

Usage examples:

# Quick smoke test (small number of steps):
python train/train_scripts/run_hsac_dex.py \
    --demo-path envs/gym-hybrid/expert_demos/expert_demo_0.npz \
    --env Moving-v0 \
    --n-timesteps 10000 \
    --device cpu \
    --save-dir logs/hsac_dex_smoke

# Full training:
python train/train_scripts/run_hsac_dex.py --demo-path /full/path/expert_demo.npz --env Moving-v0 --n-timesteps 1e7 --device cuda

The script will:
- Verify demo file is readable and print its contents/shapes
- Instantiate HSAC_DEX pointing at the demo file
- Run model.learn(...) and save the model to <save-dir>

Note: make sure the local `algs/stable-baselines3` is importable or installed (pip install -e ./algs/stable-baselines3)
      and PyTorch is installed in the environment.
"""

import argparse
import os
import sys
import time

import numpy as np
import gymnasium as gym

# Ensure local stable-baselines3 package is importable when running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SB3_PATH = os.path.join(REPO_ROOT, "algs", "stable-baselines3")
if SB3_PATH not in sys.path:
    sys.path.insert(0, SB3_PATH)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-path", required=True, help="Path to a single expert demo .npz file")
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
    print(f"Inspecting demo file: {demo_path}")
    with np.load(demo_path) as data:
        print("keys:", list(data.keys()))
        for k in data.keys():
            v = data[k]
            print(f" - {k}: shape={v.shape} dtype={v.dtype}")


def main():
    args = parse_args()
    demo_path = args.demo_path

    verify_demo(demo_path)

    # Import HSAC_DEX lazily (after verifying demo readability)
    try:
        # We prefer direct import to avoid relying on installed package
        from stable_baselines3.hsac_dex.hsac_dex import HSAC_DEX
    except Exception as e:
        print("Failed to import HSAC_DEX. Ensure stable-baselines3 local package is on PYTHONPATH or installed (pip install -e ./algs/stable-baselines3)")
        raise

    print("Creating environment:", args.env)
    env = gym.make(args.env)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = int(time.time())
    save_name = os.path.join(args.save_dir, f"hsac_dex_{args.env}_{timestamp}")

    print("Instantiating HSAC_DEX with demo_path", demo_path)
    model = HSAC_DEX(
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

    # Optionally save the replay buffer if available
    try:
        if hasattr(model, "save_replay_buffer"):
            rb_path = save_name + "_replay_buffer.pkl"
            print("Saving replay buffer to", rb_path)
            model.save_replay_buffer(rb_path)
    except Exception as e:
        print("Could not save replay buffer:", e)

    # Close env
    try:
        model.env.close()
    except Exception:
        env.close()

    print("Done.")


if __name__ == "__main__":
    main()
