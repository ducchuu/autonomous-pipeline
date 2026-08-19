"""Installation script for the 'autonomous_pipeline' python package.

Mirrors the pattern used by Isaac Lab's own external-project template
(https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html):
version is read from config/extension.toml so it never drifts out of sync
between the two files.
"""

import os
import toml

from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="autonomous_pipeline",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    packages=["autonomous_pipeline"],
    install_requires=[],
)
