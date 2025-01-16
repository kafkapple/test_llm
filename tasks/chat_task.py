from typing import Dict, Any, Optional
from .base_task import BaseTask

class ChatTask(BaseTask):
    def get_instruction(self, input_text: str) -> str:
        """Generate chat instruction"""
        return input_text  # 채팅의 경우 입력을 그대로 전달

    def process_response(self, response: str) -> str:
        """Process chat response"""
        if not response or response.isspace():
            return "죄송합니다. 응답을 생성하는데 문제가 있었습니다."
            
        # 특수 토큰 제거 및 응답 정리
        response = response.strip()
        response = response.replace("<|assistant|>", "").replace("</s>", "").strip()
        
        print(f"\nAssistant: {response}")
        return response

    def get_generation_config(self) -> Dict[str, Any]:
        """Get chat-specific generation configuration"""
        base_config = super().get_generation_config()
        
        # 채팅 전용 설정 추가
        chat_config = {
            "temperature": self.config.task.generation.temperature,
            "max_new_tokens": self.config.task.generation.max_tokens,
            "do_sample": True,
            "pad_token_id": 2,
            "eos_token_id": 2,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "min_new_tokens": 1,
            "return_full_text": False
        }
        
        return {**base_config, **chat_config} 