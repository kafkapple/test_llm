from typing import Dict, Optional, List, Union
from .base_task import BaseTask
import json
import pandas as pd

class EmotionTask(BaseTask):
    def get_instruction(self, texts: Union[str, List[str], pd.DataFrame]) -> List[str]:
        """
        텍스트 배치에 대한 instruction 생성
        
        Args:
            texts: 단일 문자열, 문자열 리스트, 또는 DataFrame
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.DataFrame):
            if 'text' not in texts.columns:
                raise ValueError("DataFrame must contain 'text' column")
            texts = texts['text'].tolist()
        
        return [
            self.config.task.template.format(
                prompt=self.config.task.prompt,
                format_instruction=self.config.task.format_instruction,
                text=text
            ) for text in texts
        ]

    def process_response(self, responses: Union[str, List[str]]) -> List[Optional[Dict]]:
        """배치 응답 처리"""
        if isinstance(responses, str):
            responses = [responses]
        
        results = []
        for i, response in enumerate(responses):
            try:
                json_str = response[response.find('{'):response.rfind('}')+1]
                result = json.loads(json_str)
                print(f"\n=== Analysis Results for Text {i+1} ===")
                print(f"Emotion: {result.get('emotion')}")
                print(f"Confidence: {result.get('confidence')}")
                print(f"Reason: {result.get('reason')}")
                print(f"Keywords: {', '.join(result.get('keywords', []))}")
                results.append(result)
            except Exception as e:
                print(f"JSON parsing error for text {i+1}: {str(e)}")
                print("Original response:", response)
                results.append(None)
        
        return results 