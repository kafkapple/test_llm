from .base_task import BaseTask
from typing import List, Dict

class ChatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.chat_history: List[Dict[str, str]] = []

    def get_instruction(self, input_text: str) -> str:
        self.chat_history.append({"role": "user", "content": input_text})
        return input_text

    def process_response(self, response: str) -> str:
        self.chat_history.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}")
        return response

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.config.tasks.chat.system_prompt}] + self.chat_history 