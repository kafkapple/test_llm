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

torchvision.disable_beta_transforms_warning()

def parse_emotion_response(response: str) -> Optional[Dict]:
    """감정 분석 응답을 JSON으로 파싱"""
    try:
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response)
        
        if not matches:
            print("응답에서 JSON을 찾을 수 없습니다.")
            return None
            
        json_str = max(matches, key=len)
        parsed = json.loads(json_str)
        
        required_fields = ["emotion", "confidence", "reason", "keywords"]
        if not all(field in parsed for field in required_fields):
            print("필수 필드가 누락되었습니다.")
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
            if cfg.task_type == "emotion" 
            else cfg.generation.temperature.default)
    
    generation_config = {
        "max_new_tokens": 256 if cfg.task_type == "emotion" else cfg.max_new_tokens,
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
    
    processor = BatchProcessor(llama_model, cfg)
    
    if hasattr(cfg.batch_processing, 'input_dir'):
        processor.process_directory(cfg.batch_processing.input_dir)
    elif hasattr(cfg.batch_processing, 'csv_file'):
        processor.process_csv(
            cfg.batch_processing.csv_file,
            cfg.batch_processing.text_column
        )
    else:
        print("입력 소스가 지정되지 않았습니다. 기본 예제를 실행합니다.")
        sample_text = (
            "In fact, the bar offered a free glass of beer to the first 100 "
            "fans to walk through the door — if they could quote a line from "
            "the song."
        )
        llama_model.generate(sample_text)

if __name__ == "__main__":
    main()
