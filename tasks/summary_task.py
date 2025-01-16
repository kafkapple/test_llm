from typing import Dict, Any
from .base_task import BaseTask

class SummaryTask(BaseTask):
    def get_instruction(self, input_text: str) -> str:
        lang = self.config.language.response
        return self.config.tasks.summary.template[lang].format(text=input_text)
    
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.config.tasks.summary.generation.max_tokens,
            "temperature": self.config.tasks.summary.generation.temperature,
            "do_sample": True,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k
        }
    
    def process_response(self, response: str) -> str:
        print("\n=== Summary ===")
        print(response)
        return response 