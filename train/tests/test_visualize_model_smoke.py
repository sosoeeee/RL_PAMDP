import os
import glob
import subprocess


def test_visualize_model_smoke(tmp_path):
    # Find latest model in logs/hsac_dex
    model_dir = os.path.join("logs", "hsac_dex")
    if not os.path.isdir(model_dir):
        raise AssertionError("Model directory not found: logs/hsac_dex")
    zips = sorted(glob.glob(os.path.join(model_dir, "*.zip")), key=os.path.getmtime)
    if not zips:
        raise AssertionError("No .zip model files found in logs/hsac_dex")
    model = zips[-1]

    out = str(tmp_path / "vis_smoke.gif")
    cmd = [
        "python",
        "train/train_scripts/visualize_trained_model.py",
        "--model",
        model,
        "--env",
        "Moving-v0",
        "--output",
        out,
        "--max-frames",
        "10",
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"visualize script failed with code {proc.returncode}")
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise AssertionError("Output GIF not created or empty")
