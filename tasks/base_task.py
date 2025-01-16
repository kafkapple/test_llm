from abc import ABC, abstractmethod
from typing import Dict, Any
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

    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        task_config = self.config.task
        
        return {
            "temperature": task_config.generation.temperature,
            "max_new_tokens": task_config.generation.max_tokens,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k,
            "do_sample": True,
            "pad_token_id": 2,  # EOS token as PAD
            "eos_token_id": 2,
        } 