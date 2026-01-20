import sys

from train_scripts.train import train
from train_scripts.tests.test_gym_with_hppo import test_gym_env
from train_scripts.tests.test_train_gym_with_hppo import test_train_gym_env
from train_scripts.tests.test_model import test_model
from train_scripts.tests.test_train_model import test_train_model


def main():
    script_name = sys.argv[1]
    # Remove script name
    del sys.argv[1]
    # Execute known script
    known_scripts = {
        "train": train,
        "test_gym_env": test_gym_env,
        "test_model": test_model,
        "test_train_gym_env": test_train_gym_env,
        "test_train_model": test_train_model
    }
    if script_name not in known_scripts.keys():
        raise ValueError(f"The script {script_name} is unknown, please use one of {known_scripts.keys()}")
    known_scripts[script_name]()


if __name__ == "__main__":
    main()
