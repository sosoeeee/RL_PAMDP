# Training Reinforcement Learning Models for Parameterized Action MDPs

This repository provides a framework for training reinforcement learning agents in environments with parameterized action spaces. It is based on a customized version of Stable-Baselines3.

## Project Structure

-   `/algs`: Contains the core reinforcement learning algorithm library. This is a customized version of `stable-baselines3` modified to support parameterized action spaces.
-   `/envs`: Place your custom `gymnasium` environments here.
-   `/train`: Includes scripts for training models, hyperparameter configurations (`hyperparams/`), and saved models (`models/`).
-   `/test`: Contains scripts for evaluating and testing trained models.

## 1. Installation

First, install the necessary packages. It is recommended to use a virtual environment.

```bash
# Install the custom RL algorithm library
pip install --user -e ./algs/stable-baselines3/ 

# Install the training scripts package
pip install --user -e ./train/ 

# Install monitoring and visualization tools
pip install tensorboard
pip install swanlab

# Install PyTorch (ensure compatibility with your CUDA version)
pip install --user torch==2.5.1 # Compiled with CUDA 12.2
```

## 2. Gym Environment Setup

Place your custom Gym environment code under the `envs/` directory. Then, install it locally in editable mode.

For example, to install the `gym-hybrid` environment:
```bash
pip install --user -e ./envs/gym-hybrid/
```
The framework will then be able to find and register the custom environment.

**Note:** After installing a new environment, make sure to import it in `train/train_scripts/import_envs.py` to make it accessible to the training scripts.

## 3. Hyperparameter Configuration

Configure environment settings and training hyperparameters in `train/hyperparams/[ALGO].yml`, where `[ALGO]` is the name of the algorithm (e.g., `hppo`).

For example, to configure the `Moving-v0` environment for the `hppo` algorithm, you can create/edit `train/hyperparams/hppo.yml`:

```yaml
Moving-v0:
  normalize: true
  n_envs: 8
  n_timesteps: !!float 1e7
  policy: 'MlpPolicy'
  batch_size: 256
  n_steps: 1024
  gamma: 0.99
  learning_rate: lin_5e-6
  clip_range: lin_0.2
  ent_coef: lin_0.025
  n_epochs: 10
  gae_lambda: 0.99
  max_grad_norm: 5
  vf_coef: 1.0
  policy_kwargs: "dict(log_std_init = 0,
                       ortho_init=False,
                       activation_fn=nn.ReLU,
                       net_arch=dict(pi=[128], vf=[128, 128, 64], di=[128, 64], co=[128, 64])
                       )"
  env_wrapper:
  - stable_baselines3.hppo.wrappers.RescaleActionWrapper:
      min_action: -1   
      max_action: 1
```

## 4. Model Training

Use the `train.py` script to start the training process. The script is a wrapper around the `train_scripts/train.py` module.

```bash
python ./train_scripts/train.py --algo hppo --env Moving-v0 --device cuda -f ./models/ -P --track --wandb-project-name "test" --vec-env "subproc" --eval-freq 4096 --eval-episodes 20 --n-eval-envs 4 --paraTag "t1"
```

### Command-line Arguments:
- `--algo`: The RL algorithm to use (e.g., `hppo`).
- `--env`: The ID of the environment to train on (e.g., `Moving-v0`).
- `--device`: The device to use for training (e.g., `cuda`, `cpu`).
- `-f`: The folder to save the trained models (e.g., `./models/`).
- `-P` or `--progress`: Display a progress bar during training.
- `--track`: Enable experiment tracking with `swanlab`.
- `--wandb-project-name`: Name of the project for `swanlab` tracking.
- `--vec-env`: The type of vectorized environment (`dummy` or `subproc` for multiprocessing).
- `--eval-freq`: Evaluate the agent every N steps.
- `--eval-episodes`: The number of episodes to run for each evaluation.
- `--n-eval-envs`: The number of parallel environments to use for evaluation.
- `--paraTag`: A tag to distinguish different training runs.

## 5. Model Testing

To test a trained model, use the `test_model.py` script. This script loads a saved model and runs it in the specified environment.

```bash
python test/test_model.py --i 1 --visualize
```

### How it works:
The `test_model.py` script performs the following steps:
1.  **Parses arguments**:
    - `--i`: The index of the model to load (corresponding to the folder `hppo/{ENV_ID}_{index}`).
    - `--visualize` or `-v`: If present, renders the environment to visualize the agent's behavior.
    - `--lastModel` or `-l`: If present, loads the last saved model instead of the best model.
2.  **Loads Hyperparameters**: It reads the same `.yml` configuration file from `train/hyperparams/` that was used for training to set up the environment correctly.
3.  **Initializes Environment**: It creates the vectorized and normalized evaluation environment.
4.  **Loads Model**: It loads the `best_model.zip` (or another specified model) from the corresponding model directory (e.g., `train/models/hppo/Moving-v0_1/`).
5.  **Runs Simulation**: It runs an infinite loop, where the agent predicts an action based on the current observation, and the environment steps forward.

## 6. Logging and Visualization

This framework uses TensorBoard and SwanLab for experiment tracking and visualization.

### TensorBoard
Training logs are automatically saved in the `train/train_scripts/runs/` directory. To view them, run the following command from the root of the project:
```bash
tensorboard --logdir train/train_scripts/runs
```

### SwanLab

This framework supports experiment tracking via SwanLab, offering both cloud-based and local visualization.

#### Cloud Synchronization

To sync your experiments to the SwanLab cloud, you first need to log in with your API key:
```bash
swanlab login
```
After logging in, simply use the `--track` flag during training. Your experiment data will be uploaded, and you can monitor the progress on the SwanLab website.

#### Local Visualization

If you prefer to view logs locally without syncing to the cloud, you can use the `watch` command. The log files are stored in `train/train_scripts/swanlog/`.
```bash
swanlab watch -l train/train_scripts/swanlog/
```
This will start a local server and provide a URL to access the dashboard in your browser.
