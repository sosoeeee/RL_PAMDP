import os
import shutil

from setuptools import setup

with open(os.path.join("train_scripts", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Copy hyperparams files for packaging
shutil.copytree("hyperparams", os.path.join("train_scripts", "hyperparams"))

long_description = """
# A custom train scripts adapted from RL Baselines3 Zoo

See https://github.com/DLR-RM/rl-baselines3-zoo
"""
install_requires = [
    "gymnasium>=0.29.1,<1.1.0",
    "tqdm",
    "rich",
    "optuna>=3.0",
]

setup(
    name="train_scripts",
    packages=["train_scripts"],
    package_data={
        "train_scripts": [
            "py.typed",
            "version.txt",
            "hyperparams/*.yml",
        ]
    },
    entry_points={"console_scripts": ["train_scripts=train_scripts.cli:main"]},
    install_requires=install_requires,
    description="A Training Framework for Stable Baselines3 Reinforcement Learning Agents",
    author="sosoeeee",
    # url="https://github.com/DLR-RM/rl-baselines3-zoo",
    # author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gymnasium openai stable baselines sb3 toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "Documentation": "https://rl-baselines3-zoo.readthedocs.io/en/master/",
        "Changelog": "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/CHANGELOG.md",
        "Stable-Baselines3": "https://github.com/DLR-RM/stable-baselines3",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

# Remove copied files after packaging
shutil.rmtree(os.path.join("train_scripts", "hyperparams"))
