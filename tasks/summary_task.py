from .base_task import BaseTask

class SummaryTask(BaseTask):
    def get_instruction(self, input_text: str) -> str:
        return self.config.tasks.summary.template.format(text=input_text)

    def process_response(self, response: str) -> str:
        print(f"===Summary:\n{response}")
        return response 