from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
from omegaconf import DictConfig
from pathlib import Path

class ModelFactory:
    """Factory class for managing various LLM models"""
    
    @staticmethod
    def create_quantization_config(
        precision: str
    ) -> Optional[BitsAndBytesConfig]:
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
    def load_model(
        model_path: str,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        """Load model"""
        kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,  # Optimize memory usage
            "trust_remote_code": True,
        }
        
        try:
            print("Attempting to load model in float16...")
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                **kwargs
            )
        except Exception as e:
            print(f"Failed to load in float16: {str(e)}")
            try:
                print("Retrying with float32...")
                kwargs["torch_dtype"] = torch.float32
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **kwargs
                )
            except Exception as e:
                print(f"Failed with float32: {str(e)}")
                try:
                    print("Final attempt using CPU...")
                    kwargs["device_map"] = "cpu"
                    return AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **kwargs
                    )
                except Exception as e:
                    print(f"All attempts failed: {str(e)}")
                    raise

    @staticmethod
    def create_model_and_tokenizer(
        cfg: DictConfig
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Create model and tokenizer"""
        model_name = cfg.model.name
        
        # Set model save path
        save_path = Path(cfg.model.save_path) / model_name.split('/')[-1]
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        if cfg.model.cache.enabled:
            cache_dir = save_path
        else:
            cache_dir = None
        
        print(f"Model save path: {save_path}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                force_download=cfg.model.force_download
            )
        except Exception as e:
            print(f"Failed to load tokenizer: {str(e)}")
            # Retry from local path
            if save_path.exists():
                print(f"Attempting to load tokenizer from local path: {save_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(save_path),
                    trust_remote_code=True
                )
            else:
                raise
        
        # Set default chat template
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
        
        # Load model
        try:
            if cfg.quantization.enabled:
                if cfg.quantization.precision == "int8":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        load_in_8bit=True,
                        trust_remote_code=True,
                        device_map="auto",
                        force_download=cfg.model.force_download
                    )
                elif cfg.quantization.precision == "int4":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        load_in_4bit=True,
                        trust_remote_code=True,
                        device_map="auto",
                        force_download=cfg.model.force_download
                    )
                else:
                    raise ValueError(
                        f"Unsupported quantization precision: {cfg.quantization.precision}"
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    device_map="auto",
                    force_download=cfg.model.force_download
                )
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            # Retry from local path
            if save_path.exists():
                print(f"Attempting to load model from local path: {save_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    str(save_path),
                    trust_remote_code=True,
                    device_map="auto"
                )
            else:
                raise
        
        return model, tokenizer 