from typing import Any, Optional
from omegaconf import DictConfig
from tasks.factory import TaskFactory
from models.model_factory import ModelFactory

class LLMPipeline:
    """Pipeline for managing the flow of LLM tasks"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task = TaskFactory.create_task(config)
        self.model = self._create_model()
    
    def _create_model(self):
        """Create appropriate model based on config"""
        model_type = self.config.model.get("type", "llama")
        model = ModelFactory.create_model(model_type, self.config)
        return model

    def run(self, input_text: str = None):
        """Run the pipeline based on task type"""
        if self.config.task_type == "chat":
            self._run_chat_mode()
        else:
            self._run_single_task_mode(input_text)

    def _run_chat_mode(self):
        """Run interactive chat mode"""
        print("\n=== Starting chat mode ===")
        lang = self.config.language.response
        exit_keywords = self.config.tasks.chat.exit_keywords[lang]
        print(f"You can exit using these keywords: {', '.join(exit_keywords)}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in exit_keywords:
                print("\n=== Ending chat ===")
                break
            self.process(user_input)

    def _run_single_task_mode(self, input_text: str):
        """Run single task mode"""
        if not input_text:
            raise ValueError("No input text provided for single task mode")
        self.process(input_text)

    def process(self, input_text: str) -> Any:
        """Process input through the pipeline"""
        instruction = self.task.get_instruction(input_text)
        generation_config = self.task.get_generation_config()
        response = self.model.generate(instruction, generation_config)
        return self.task.process_response(response) 