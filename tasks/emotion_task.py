import json
from typing import Dict, Any, Optional
from .base_task import BaseTask

class EmotionTask(BaseTask):
    def get_instruction(self, input_text: str) -> str:
        return self.config.tasks.emotion.template.format(
            prompt=self.config.tasks.emotion.prompt,
            format_instruction=self.config.tasks.emotion.format_instruction,
            text=input_text
        )
    
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.config.tasks.emotion.generation.max_tokens,
            "temperature": self.config.tasks.emotion.generation.temperature,
            "do_sample": True,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k
        }
    
    def process_response(self, response: str) -> Optional[Dict]:
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            result = json.loads(json_str)
            print("\n=== Emotion Analysis Results ===")
            print(f"Emotion: {result.get('emotion')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Reason: {result.get('reason')}")
            print(f"Keywords: {', '.join(result.get('keywords', []))}")
            return result
        except Exception as e:
            print(f"\n=== JSON parsing failed: {str(e)} ===")
            print(response)
            return None 