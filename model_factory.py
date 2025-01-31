from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
from omegaconf import DictConfig

class ModelFactory:
    """다양한 LLM 모델을 생성하고 관리하는 팩토리 클래스"""
    
    @staticmethod
    def create_quantization_config(
        precision: str
    ) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 생성"""
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
    def load_model(
        model_path: str,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        """모델 로딩"""
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,  # 메모리 사용 최적화
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

    @staticmethod
    def create_model_and_tokenizer(
        cfg: DictConfig
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저 생성"""
        model_name = cfg.model.name
        cache_dir = cfg.model.save_path if cfg.model.cache.enabled else None
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # 기본 chat template 설정
        if tokenizer.chat_template is None:
            template = (
                "{% if not add_generation_prompt is defined %}"
                "{% set add_generation_prompt = false %}{% endif %}"
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + "
                "message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}{% endif %}"
            )
            tokenizer.chat_template = template
        
        # 양자화 설정
        if cfg.quantization.enabled:
            if cfg.quantization.precision == "int8":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    load_in_8bit=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
            elif cfg.quantization.precision == "int4":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    load_in_4bit=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                raise ValueError(
                    f"지원하지 않는 양자화 정밀도: {cfg.quantization.precision}"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map="auto"
            )
        
        return model, tokenizer 