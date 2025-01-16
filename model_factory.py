from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from omegaconf import DictConfig

class ModelFactory:
    """Factory class for creating and managing various LLM models"""
    
    @staticmethod
    def create_quantization_config(precision: str) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration"""
        if precision not in ["int4", "int8"]:
            return None
            
        return BitsAndBytesConfig(
            load_in_8bit=precision == "int8",
            load_in_4bit=precision == "int4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    @staticmethod
    def load_model(model_path: str, quantization_config: Optional[BitsAndBytesConfig] = None):
        """Load model"""
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,  # Optimize memory usage
            "trust_remote_code": True,
        }
        
        try:
            print("float16으로 모델 로딩 시도...")
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                **kwargs
            )
        except Exception as e:
            print(f"float16 로딩 실패: {str(e)}")
            try:
                print("float32로 다시 시도...")
                kwargs["torch_dtype"] = torch.float32
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **kwargs
                )
            except Exception as e:
                print(f"float32로도 실패: {str(e)}")
                try:
                    print("CPU로 마지막 시도...")
                    kwargs["device_map"] = "cpu"
                    return AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **kwargs
                    )
                except Exception as e:
                    print(f"모든 시도 실패: {str(e)}")
                    raise

    @classmethod
    def create_model_and_tokenizer(cls, cfg: DictConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Create or load model and tokenizer"""
        model_dir = Path(cfg.model.save_path) / cfg.model.name.split('/')[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 양자화 설정
        bnb_config = cls.create_quantization_config(cfg.quantization.precision) if cfg.quantization.enabled else None
        
        if (model_dir / "model").exists() and not cfg.model.force_download:
            print(f"Loading model from {model_dir}")
            model = cls.load_model(str(model_dir / "model"), bnb_config)
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
        else:
            print(f"Downloading model {cfg.model.name}...")
            model = cls.load_model(cfg.model.name, bnb_config)
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
            
            # 모델과 토크나이저 저장
            model.save_pretrained(str(model_dir / "model"))
            tokenizer.save_pretrained(str(model_dir / "tokenizer"))
            
        return model, tokenizer 