import hydra
import re
import json
import torch
import torchvision
from typing import Optional, Dict
from pytorch_lightning import seed_everything, LightningModule
from transformers import AutoTokenizer
from omegaconf import DictConfig
from templates import get_template
from model_factory import ModelFactory
from batch_processor import BatchProcessor
from pathlib import Path

torchvision.disable_beta_transforms_warning()

def parse_emotion_response(response: str) -> Optional[Dict]:
    """감정 분석 응답을 JSON으로 파싱"""
    try:
        # 중첩된 중괄호를 포함한 JSON 블록 찾기
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if not matches:
            print("응답에서 JSON을 찾을 수 없습니다.")
            return None
            
        # 가장 긴 JSON 문자열 선택
        json_str = max(matches, key=len)
        parsed = json.loads(json_str)
        
        # 필수 필드 검증
        required_fields = ["emotion", "confidence", "reason", "keywords"]
        if not all(field in parsed for field in required_fields):
            print("필수 필드가 누락되었습니다.")
            return None
            
        # 감정 값 검증
        valid_emotions_ko = ["기쁨", "분노", "슬픔", "놀람", "혐오", "두려움", "중립"]
        valid_emotions_en = ["happy", "angry", "sad", "surprise", "disgust", "fear", "neutral"]
        
        emotion = parsed["emotion"]
        if not (emotion in valid_emotions_ko or emotion in valid_emotions_en):
            print(f"잘못된 감정 값: {emotion}")
            return None
            
        # confidence 값 검증
        confidence = float(parsed["confidence"])
        if not (0 <= confidence <= 1):
            print(f"잘못된 confidence 값: {confidence}")
            return None
            
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {str(e)}")
        return None
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        return None

def get_instruction(cfg: DictConfig, input_text: str) -> str:
    """템플릿에 따른 지시문 생성"""
    template = get_template(cfg.task_type, cfg.language)
    if template is None:
        raise ValueError(f"지원하지 않는 태스크 타입입니다: {cfg.task_type}")
    
    if cfg.task_type == "query":
        return template.format(domain=cfg.domain)
    return template.format(text=input_text)

def generate_prompt(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    model,
    input_text: str
) -> str:
    """프롬프트 생성 및 모델 응답 생성"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 언어에 따른 프롬프트 선택
    prompt = cfg.prompt.get(cfg.language, cfg.prompt.korean)
    instruction = get_instruction(cfg, input_text)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instruction}
    ]
    
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    input_ids = encoded.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    temp = (cfg.generation.temperature.emotion 
            if "emotion" in cfg.task_type
            else cfg.generation.temperature.default)
    print(temp)
    generation_config = {
        "max_new_tokens": 256 if "emotion" in cfg.task_type else cfg.max_new_tokens,
        "temperature": temp,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "top_p": cfg.generation.top_p,
        "top_k": cfg.generation.top_k
    }
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    
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
    
    return response

class LLamaLightningModel(LightningModule):
    """LLaMA 모델을 위한 Lightning 모듈"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(
            cfg
        )
        print(
            f"===Task Type: {cfg.task_type}\n"
            f"===Model name: {cfg.model.name}\n"
            f"===Precision: {cfg.quantization.precision}\n"
            f"===Quantization: {cfg.quantization.enabled}"
        )

    def generate(self, input_text: str) -> str:
        """주어진 입력 텍스트에 대한 응답 생성"""
        return generate_prompt(self.cfg, self.tokenizer, self.model, input_text)

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 실행 함수"""
    seed_everything(cfg.seed, workers=True)
    llama_model = LLamaLightningModel(cfg)
    
    if cfg.debug.enabled:
        print("===Debug Mode===")
        print(cfg)
        sample_text = "Wow! I'm so happy! "
        llama_model.generate(sample_text)
    else:
        processor = BatchProcessor(llama_model, cfg)

        # input_dir이 None이 아니고 실제 존재하는 경우에만 디렉토리 처리
        if (hasattr(cfg.batch_processing, 'input_dir') and 
            cfg.batch_processing.input_dir is not None and 
            Path(cfg.batch_processing.input_dir).exists()):
            processor.process_directory(cfg.batch_processing.input_dir)
        
        # CSV 파일 처리
        elif (hasattr(cfg.batch_processing, 'csv_file') and 
            Path(cfg.batch_processing.csv_file).exists()):
            processor.process_csv(
                cfg.batch_processing.csv_file,
                cfg.batch_processing.text_column
            )
        
        else:
            print("Text is not provided.")
            sample_text = (
                "In fact, the bar offered a free glass of beer to the first 100 "
                "fans to walk through the door — if they could quote a line from "
                "the song."
            )
            llama_model.generate(sample_text)

if __name__ == "__main__":
    main()
