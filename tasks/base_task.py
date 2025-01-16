from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from omegaconf import DictConfig

class BaseTask(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def get_instruction(self, input_text: str) -> str:
        """Generate instruction for the task"""
        pass

    @abstractmethod
    def process_response(self, response: str) -> Any:
        """Process model response"""
        pass

    def get_generation_config(self) -> Dict:
        """Get generation configuration for the task"""
        return {
            "max_new_tokens": self.config.tasks[self.config.task_type].generation.max_tokens,
            "temperature": self.config.tasks[self.config.task_type].generation.temperature,
            "do_sample": True,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k
        } 