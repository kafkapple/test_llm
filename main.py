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
import time

torchvision.disable_beta_transforms_warning()

def parse_emotion_response(response: str) -> Optional[Dict]:
    """Parse emotion analysis response into JSON"""
    try:
        print(f"===Response: {response}")
        # Find JSON block including nested braces
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if not matches:
            print("No JSON found in response.")
            return None
            
        # Select longest JSON string
        json_str = max(matches, key=len)
        parsed = json.loads(json_str.strip())
        
        # Validate required fields
        required_fields = ["emotion", "confidence", "reason", "keywords", "arousal", "valence"]
        if not all(field in parsed for field in required_fields):
            print("Missing required fields.")
            return None
            
        # Validate emotion value
        valid_emotions_ko = ["기쁨", "분노", "슬픔", "놀람", "혐오", "두려움", "중립"]
        valid_emotions_en = ["happy", "angry", "sad", "surprise", "disgust", "fear", "neutral"]
        
        emotion = parsed["emotion"].lower()
        if not (emotion in [e.lower() for e in valid_emotions_ko] or 
                emotion in [e.lower() for e in valid_emotions_en]):
            print(f"Invalid emotion value: {emotion}")
            return None
            
        # Validate numeric fields
        for field in ["confidence", "arousal", "valence"]:
            try:
                value = float(parsed[field])
                if not (0 <= value <= 1):
                    print(f"Invalid {field} value: {value}")
                    return None
            except (ValueError, TypeError):
                print(f"Invalid {field} format: {parsed[field]}")
                return None
        
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

def get_instruction(cfg: DictConfig, input_text: str) -> str:
    """Generate instruction based on template"""
    template = get_template(cfg.task_type, cfg.language)
    if template is None:
        raise ValueError(f"Unsupported task type: {cfg.task_type}")
    
    if cfg.task_type == "query":
        return template.format(domain=cfg.domain)
    return template.format(text=input_text)

def generate_prompt(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    model,
    input_text: str
) -> str:
    """Generate prompt and model response"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Select prompt based on language
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
            print("===Analysis Result:")
            print(f"Emotion: {parsed_response.get('emotion')}")
            print(f"Arousal: {parsed_response.get('arousal')}")
            print(f"Valence: {parsed_response.get('valence')}")
            print(f"Confidence: {parsed_response.get('confidence')}")
            print(f"Reason: {parsed_response.get('reason')}")
            print(f"Keywords: {', '.join(parsed_response.get('keywords', []))}")
        else:
            print("JSON parsing failed. Original response:")
            print(response)
    else:
        print(f"===Input: {instruction}\n===Generated Response:\n{response}")
    
    return response

class LLamaLightningModel(LightningModule):
    """Lightning module for LLaMA model"""
    
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
        """Generate response for given input text"""
        return generate_prompt(self.cfg, self.tokenizer, self.model, input_text)

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main execution function"""
    seed_everything(cfg.seed, workers=True)
    llama_model = LLamaLightningModel(cfg)
    
    start_time = time.time()
    if cfg.debug.enabled:
        print("===Debug Mode===")
        print(cfg)
        sample_text = "Wow! I'm so happy! "
        llama_model.generate(sample_text)
    else:
        processor = BatchProcessor(llama_model, cfg)

        # Process directory only if input_dir exists
        if (hasattr(cfg.batch_processing, 'input_dir') and 
            cfg.batch_processing.input_dir is not None and 
            Path(cfg.batch_processing.input_dir).exists()):
            processor.process_directory(cfg.batch_processing.input_dir)
        
        # Process CSV file
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
    end_time = time.time()
    print(f"===Time: {end_time - start_time}")

if __name__ == "__main__":
    main()

