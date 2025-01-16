from typing import Type
from .base_task import BaseTask
from .emotion_task import EmotionTask
from .summary_task import SummaryTask
from .chat_task import ChatTask
from omegaconf import DictConfig

class TaskFactory:
    _tasks = {
        "emotion": EmotionTask,
        "summary": SummaryTask,
        "chat": ChatTask
    }

    @classmethod
    def create_task(cls, config: DictConfig) -> BaseTask:
        task_type = config.task_type
        if task_type not in cls._tasks:
            raise ValueError(f"Unsupported task type: {task_type}")
        return cls._tasks[task_type](config) 