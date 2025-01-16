from typing import Dict, Optional
from .base_task import BaseTask
import json

class EmotionTask(BaseTask):
    def get_instruction(self, input_text: str) -> str:
        return self.config.tasks.emotion.template.format(
            prompt=self.config.tasks.emotion.prompt,
            format_instruction=self.config.tasks.emotion.format_instruction,
            text=input_text
        )

    def process_response(self, response: str) -> Optional[Dict]:
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            result = json.loads(json_str)
            print(f"===Analysis Results:")
            print(f"Emotion: {result.get('emotion')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Reason: {result.get('reason')}")
            print(f"Keywords: {', '.join(result.get('keywords', []))}")
            return result
        except Exception as e:
            print(f"JSON parsing error: {str(e)}")
            print("Original response:", response)
            return None 