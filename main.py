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
        lang = cfg.language.response
        exit_keywords = cfg.tasks.chat.exit_keywords[lang]
        print(f"You can exit using these keywords: {', '.join(exit_keywords)}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in exit_keywords:
                print("\n=== Ending chat ===")
                break
            pipeline.process(user_input)
    else:
        # Single task mode
        input_text = """topology manifold
homeomorphism
A unified theory for the computational and mechanistic origins of grid cells
"""
        input_text = """I can't stand it. How can I get rid of it?"""
        pipeline.process(input_text)

if __name__ == "__main__":
    main()
