from setuptools import find_packages, setup

setup(
    name="anchor-rl",
    version="0.1.0",
    python_requires=">=3.10,<3.11",
    packages=find_packages(include=["wsrl", "wsrl.*", "experiments", "experiments.*"]),
    description="ANCHOR: KL-Anchored Adaptation for Stable Offline-to-Online RL",
)
