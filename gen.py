import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import seed_everything, LightningModule
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path
import torchvision
torchvision.disable_beta_transforms_warning()
from model_factory import ModelFactory
import json
from typing import Optional, Dict

def parse_emotion_response(response: str) -> Optional[Dict]:
    """감정 분석 응답을 JSON으로 파싱"""
    try:
        # JSON 부분만 추출 (중괄호로 둘러싸인 부분)
        json_str = response[response.find('{'):response.rfind('}')+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON 파싱 오류: {str(e)}")
        return None

def generate_prompt(cfg: DictConfig, tokenizer, model, input_text):
    """프롬프트 생성 및 모델 응답 생성"""
    def get_instruction(cfg, input_text=input_text):
        if cfg.task_type == "summary":
            return cfg.summary_template.format(text=input_text)
        elif cfg.task_type == "emotion":
            return cfg.emotion_template.format(text=input_text)
        else:  # query
            return cfg.prompt_template.format(domain=cfg.domain)
    
    prompt = cfg.prompt_template.format(domain=cfg.domain) if cfg.task_type == "query" else cfg.prompt
    instruction = get_instruction(cfg)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    generation_config = {
        "max_new_tokens": 100 if cfg.task_type == "emotion" else cfg.max_new_tokens,
        "temperature": 0.1 if cfg.task_type == "emotion" else 0.3,
        "do_sample": True,
        "eos_token_id": [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        "top_p": 0.9,
        "top_k": 5 if cfg.task_type == "emotion" else 10
    }
    
    outputs = model.generate(input_ids, **generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # 감정 분석인 경우 JSON 파싱 시도
    if cfg.task_type == "emotion":
        parsed_response = parse_emotion_response(response)
        if parsed_response:
            print(f"===Input: {instruction}")
            print("===분석 결과:")
            print(f"감정: {parsed_response.get('emotion')}")
            print(f"확신도: {parsed_response.get('confidence')}")
            print(f"근거: {parsed_response.get('reason')}")
            print(f"주요 단어: {', '.join(parsed_response.get('keywords', []))}")
        else:
            print("JSON 파싱 실패. 원본 응답:")
            print(response)
    else:
        print(f"===Input: {instruction}\n===Generated Response:\n{response}")

class LLamaLightningModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(cfg)
        print(f"===Task Type: {cfg.task_type}\n===Model name: {cfg.model.name}\n===Precision: {cfg.quantization.precision}\n===Quantization: {cfg.quantization.enabled}")

    def generate(self, input_text: str):
        """
        Generate response for a specific domain.
        """

        generate_prompt(self.cfg, self.tokenizer, self.model, input_text)
       

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Seed for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Initialize model
    llama_model = LLamaLightningModel(cfg)
    input_text = """BGE M3-Embedding의 배경과 필요성기존 임베딩 모델의 한계언어적 제한: 대부분의 모델은 영어에 최적화되어 있어 다른 언어에서는 성능이 저하됩니다.단일 검색 방식: 하나의 검색 방식에 특화된 모델은 다양한 검색 작업에서 유연성이 부족합니다.긴 문서 처리 미흡: 긴 문서를 처리하거나 분석하는 데 어려움이 있습니다.BGE M3-Embedding은 이러한 문제를 해결하기 위해 설계된 다목적 임베딩 모델입니다.BGE M3-Embedding의 주요 특징1. 다국어 지원 (Multi-Linguality)100개 이상의 언어를 지원하며, 한국어를 포함한 다양한 언어의 문서를 효율적으로 검색할 수 있습니다.교차 언어 검색: 영어로 질문하고 한국어 문서를 검색하거나 그 반대도 가능합니다.2. 다기능 검색 (Multi-Functionality)Dense Retrieval, Sparse Retrieval, Multi-Vector Retrieval을 모두 지원하며, 하나의 모델에서 통합적으로 작동합니다.Self-Knowledge Distillation 기술을 통해 각 검색 방식의 점수를 통합하여 최적의 검색 결과를 제공합니다.3. 다중 입력 길이 지원 (Multi-Granularity)최대 8192 토큰 길이의 문서도 처리 가능하도록 설계되었습니다.짧은 문장과 긴 문단 단위 모두에서 뛰어난 성능을 발휘합니다.효율적인 배칭 전략을 적용해 긴 시퀀스 처리에서도 높은 효율성을 보장합니다.
검색 방식의 세부 설명Dense Retrieval사전 훈련된 인코더(예: BERT)를 활용하여 질문과 문서의 [CLS] 토큰 임베딩으로 유사도를 계산합니다.Sparse RetrievalBM25와 같은 기존 방식의 연장선에서, 단어 토큰에 초점을 맞춘 검색 방식을 적용합니다.각 토큰의 임베딩 값을 활용해 더 정밀한 검색 결과를 제공합니다.Multi-Vector RetrievalDense 방식과 달리 모든 토큰 임베딩을 활용하거나, 질문 및 문서를 변형해 복수의 벡터를 생성합니다.예: 질문에 대한 다양한 유사 질문을 생성하거나, 문서를 요약하여 여러 벡터를 생성해 검색의 다양성과 정확도를 높입니다.
기술적 혁신1. Self-Knowledge DistillationDense, Sparse, Multi-Vector Retrieval에서 나온 점수를 결합해 모델 학습을 최적화합니다.2. 효율적인 배칭 전략긴 문서 및 다양한 입력 길이를 처리하기 위해 대규모 배치 크기를 효율적으로 관리하는 전략을 적용했습니다.3. 종합적인 데이터 큐레이션Wikipedia, 뉴스, 번역 데이터셋 등 다양한 출처에서 데이터를 수집하고, 교차 언어 검색을 지원하기 위해 번역 데이터를 활용했습니다.GPT-3.5를 활용해 긴 문서와 질문 데이터를 생성하며 데이터 증강을 수행했습니다.

"""
    #input_text = "난 너무 화가나. 알아?"
    # Generate response
    llama_model.generate(input_text)

# # Register configuration schema
# cs = ConfigStore.instance()
# cs.store(name="config", node=DictConfig)

if __name__ == "__main__":
    main()
