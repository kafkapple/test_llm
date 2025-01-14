from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer
import torch
from konlpy.tag import Mecab
import spacy
from dataclasses import dataclass

@dataclass
class PreprocessingConfig:
    """전처리 설정을 위한 데이터 클래스"""
    max_length: int = 512
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"
    do_morphological_analysis: bool = False

class TextPreprocessor(ABC):
    """텍스트 전처리를 위한 추상 기본 클래스"""
    
    def __init__(self, model_name: str, config: Optional[PreprocessingConfig] = None):
        self.model_name = model_name
        self.config = config or PreprocessingConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    @abstractmethod
    def analyze_morphology(self, text: str) -> List[tuple]:
        """형태소/품사 분석을 수행하는 추상 메서드"""
        pass
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """공통 토크나이징 로직"""
        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors
        )
    
    def preprocess(self, text: str) -> Dict[str, Union[str, torch.Tensor, List]]:
        """공통 전처리 파이프라인"""
        # 기본 텍스트 정제
        text = self._clean_text(text)
        
        # 토크나이징
        tokenized = self.tokenize(text)
        
        result = {
            "original_text": text,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
        
        # 형태소/품사 분석 추가 (설정된 경우)
        if self.config.do_morphological_analysis:
            result["morphological_analysis"] = self.analyze_morphology(text)
        
        return result
    
    def batch_preprocess(self, texts: List[str]) -> List[Dict]:
        """배치 전처리"""
        return [self.preprocess(text) for text in texts]
    
    def _clean_text(self, text: str) -> str:
        """기본 텍스트 정제 로직"""
        return text.strip()

class KoreanPreprocessor(TextPreprocessor):
    """한국어 텍스트 전처리기"""
    
    def __init__(self, model_name: str = "beomi/llama-2-ko-7b", 
                 config: Optional[PreprocessingConfig] = None):
        super().__init__(model_name, config)
        self.mecab = Mecab()
    
    def analyze_morphology(self, text: str) -> List[tuple]:
        """Mecab을 사용한 형태소 분석"""
        return self.mecab.pos(text)
    
    def _clean_text(self, text: str) -> str:
        """한국어 특화 텍스트 정제"""
        text = super()._clean_text(text)
        # 한국어 특화 정제 규칙 추가 가능
        return text

class EnglishPreprocessor(TextPreprocessor):
    """영어 텍스트 전처리기"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf",
                 config: Optional[PreprocessingConfig] = None):
        super().__init__(model_name, config)
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_morphology(self, text: str) -> List[tuple]:
        """spaCy를 사용한 품사 태깅"""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def _clean_text(self, text: str) -> str:
        """영어 특화 텍스트 정제"""
        text = super()._clean_text(text)
        # 영어 특화 정제 규칙 추가 가능
        return text

# 언어 팩토리 클래스
class TextPreprocessorFactory:
    """전처리기 생성을 위한 팩토리 클래스"""
    
    @staticmethod
    def create_preprocessor(language: str, 
                          model_name: Optional[str] = None,
                          config: Optional[PreprocessingConfig] = None) -> TextPreprocessor:
        if language.lower() == "korean" or language.lower() == "ko":
            model_name = model_name or "beomi/llama-2-ko-7b"
            return KoreanPreprocessor(model_name, config)
        elif language.lower() == "english" or language.lower() == "en":
            model_name = model_name or "meta-llama/Llama-2-7b-hf"
            return EnglishPreprocessor(model_name, config)
        else:
            raise ValueError(f"Unsupported language: {language}")