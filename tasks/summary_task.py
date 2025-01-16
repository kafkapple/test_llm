from typing import Dict, Any, List
from .base_task import BaseTask
import re

class SummaryTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.max_chunk_tokens = 1000  # 청크당 최대 토큰 수
        self.overlap_tokens = 100      # 오버랩 토큰 수
        
    def split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 문장 구분자: 마침표, 느낌표, 물음표 뒤에 공백이나 줄바꿈이 오는 경우
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks_by_tokens(self, sentences: List[str], tokenizer) -> List[str]:
        """문장들을 토큰 수 기준으로 청크로 분할"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # 문장의 토큰 수 계산
            tokens = len(tokenizer.encode(sentence))
            
            # 현재 청크가 비어있거나, 문장을 추가해도 최대 토큰 수를 넘지 않는 경우
            if not current_chunk or current_length + tokens <= self.max_chunk_tokens:
                current_chunk.append(sentence)
                current_length += tokens
            else:
                # 현재 청크를 저장하고 새로운 청크 시작
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = tokens
        
        # 마지막 청크 처리
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        print(f"\n=== Split into {len(chunks)} chunks (token-based) ===")
        return chunks

    def get_instruction(self, input_text: str) -> str:
        """Map-Reduce 요약 생성"""
        # 1. 문장 단위로 분할
        sentences = self.split_into_sentences(input_text)
        print(f"Total sentences: {len(sentences)}")
        
        # 2. 토큰 기준으로 청크 생성
        from models.model_factory import ModelFactory  # 토크나이저 접근용
        tokenizer = ModelFactory.get_tokenizer()
        chunks = self.create_chunks_by_tokens(sentences, tokenizer)
        
        if len(chunks) == 1:
            return self.config.task.template.format(text=input_text)
            
        # Map: 각 청크 요약
        print("\n=== Generating chunk summaries ===")
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            instruction = f"Summarize this part ({i}/{len(chunks)}):\n{chunk}"
            chunk_summaries.append(instruction)
            
        # Reduce: 최종 요약
        combined_summary = "\n\n".join(chunk_summaries)
        final_instruction = "Combine these summaries into a coherent summary with proper structure:\n" + combined_summary
        
        return final_instruction

    def process_response(self, response: str) -> str:
        if not response or response.isspace():
            return "죄송합니다. 요약을 생성하는데 문제가 있었습니다."
            
        print("\n=== Final Summary ===")
        print(response)
        return response 