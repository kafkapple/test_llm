from typing import Any
from omegaconf import DictConfig
from tasks.factory import TaskFactory
from models.llama_model import LlamaModel

class LLMPipeline:
    def __init__(self, config: DictConfig):
        self.config = config
        self.task = TaskFactory.create_task(config)
        self.model = LlamaModel(config)
        self.model.load()

    def process(self, input_text: str) -> Any:
        instruction = self.task.get_instruction(input_text)
        generation_config = self.task.get_generation_config()
        response = self.model.generate(instruction, generation_config)
        return self.task.process_response(response) 