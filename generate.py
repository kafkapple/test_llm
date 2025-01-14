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

def load_model(model_path: str, quantization_config: BitsAndBytesConfig = None):
    """모델 로딩을 위한 공통 함수"""
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

def get_model_and_tokenizer(cfg: DictConfig):
    """Hugging Face 모델 및 토크나이저 생성 또는 로드"""
    model_dir = Path(cfg.model.save_path) / cfg.model.name.split('/')[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=(cfg.precision == "int8"),
        load_in_4bit=(cfg.precision == "int4")
    ) if cfg.quantization else None
    
    if (model_dir / "model").exists() and not cfg.model.force_download:
        print(f"Loading model from {model_dir}")
        model = load_model(str(model_dir / "model"), bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    else:
        print(f"Downloading model {cfg.model.name}...")
        model = load_model(cfg.model.name, bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        
        # 모델과 토크나이저 저장
        model.save_pretrained(str(model_dir / "model"))
        tokenizer.save_pretrained(str(model_dir / "tokenizer"))
        
    return model, tokenizer

def generate_prompt(cfg: DictConfig, tokenizer, model):
    """프롬프트 생성 및 모델 응답 생성"""
    def get_instruction(cfg, input_text=""):
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
        "max_new_tokens": 10 if cfg.task_type == "emotion" else cfg.max_new_tokens,
        "temperature": 0.3 if cfg.task_type == "emotion" else 0.6,
        "do_sample": True,
        "eos_token_id": [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        "top_p": 0.9,
        "top_k": 10
    }
    
    outputs = model.generate(input_ids, **generation_config)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    print(f"===Input: {instruction}\n===Generated Response:\n{response}")

class LLamaLightningModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model, self.tokenizer = get_model_and_tokenizer(cfg)#download_and_quantize_model(cfg)

    def generate(self, domain: str):
        """
        Generate response for a specific domain.
        """
        self.cfg.domain = domain
        generate_prompt(self.cfg, self.tokenizer, self.model)

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Seed for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Initialize model
    llama_model = LLamaLightningModel(cfg)

    # Generate response
    llama_model.generate(cfg.domain)

# # Register configuration schema
# cs = ConfigStore.instance()
# cs.store(name="config", node=DictConfig)

if __name__ == "__main__":
    main()
