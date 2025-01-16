from abc import ABC, abstractmethod
from typing import Any, Dict
from omegaconf import DictConfig

class BaseTask(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
    
    @abstractmethod
    def get_instruction(self, input_text: str) -> str:
        """Get instruction for the task"""
        pass
        
    @abstractmethod
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        pass
        
    @abstractmethod
    def process_response(self, response: str) -> Any:
        """Process model response"""
        pass 