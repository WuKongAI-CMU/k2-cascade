"""k2-tasks-env: four small, verifiable coding tasks packaged as a reward-hacking-resistant RL environment."""
from .env import CodingTaskEnv, Observation, StepResult
from .grader import Grade, grade_workspace
from .tasks import TASKS, Task, get_task, list_tasks

__all__ = ["CodingTaskEnv", "Observation", "StepResult", "Grade", "grade_workspace", "TASKS", "Task", "get_task", "list_tasks"]
__version__ = "0.1.0"
