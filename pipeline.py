from typing import Any, Union, List
import pandas as pd
from omegaconf import DictConfig
from tasks.factory import TaskFactory
from models.llama_model import LlamaModel

class LLMPipeline:
    def __init__(self, config: DictConfig):
        self.config = config
        self.task = TaskFactory.create_task(config)
        self.model = LlamaModel(config)
        self.model.load()

    def process(self, input_data: Union[str, List[str], pd.DataFrame]) -> Any:
        """
        Process single or batch inputs
        
        Args:
            input_data: 단일 문자열, 문자열 리스트, 또는 DataFrame
        """
        instructions = self.task.get_instruction(input_data)
        generation_config = self.task.get_generation_config()
        responses = self.model.generate(instructions, generation_config)
        return self.task.process_response(responses) 