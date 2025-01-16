import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pipeline import LLMPipeline

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    seed_everything(cfg.seed, workers=True)

    # Initialize pipeline
    pipeline = LLMPipeline(cfg)

    if cfg.task_type == "chat":
        # Chat mode
        print("\n=== Starting chat mode ===")
        print(f"You can exit using these keywords: {', '.join(cfg.tasks.chat.exit_keywords)}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in cfg.tasks.chat.exit_keywords:
                print("\n=== Ending chat ===")
                break
            pipeline.process(user_input)
    else:
        # Single task mode
        input_text = """오늘 정신과, 고용센터 상담에서는 같은 주제가 반복되었다. 완벽주의 내려놓기, 그리고 감정을 끄적여서 풀어내기."""
        pipeline.process(input_text)

if __name__ == "__main__":
    main()
