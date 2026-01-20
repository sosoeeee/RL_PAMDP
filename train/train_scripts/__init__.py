import os

# isort: off

import train_scripts.gym_patches  # noqa: F401

# isort: on

from train_scripts.utils import (
    ALGOS,
    get_latest_run_id,
    get_wrapper_class,
    linear_schedule,
)

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "ALGOS",
    "get_latest_run_id",
    "get_wrapper_class",
    "linear_schedule",
]
