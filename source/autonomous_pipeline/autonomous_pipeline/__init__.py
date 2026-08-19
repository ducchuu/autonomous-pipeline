"""autonomous_pipeline: this project's Isaac Lab extension package.

Registers custom Direct RL environments defined under `tasks/`. Import this
package (or any submodule) before `gym.make(...)` so the gym.register calls
in tasks/direct/drone_navigation/__init__.py have executed.
"""

from . import tasks  # noqa: F401
