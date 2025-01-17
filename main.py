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

    # Run pipeline with appropriate input
    if cfg.task_type != "chat":
        # Single task mode
        input_text = """topology manifold
A unified theory for the computational and mechanistic origins of grid cells
저도 이 논문 보면서 topology에 대한 배경지식의 필요성을 느꼈어요"""
        #input_text = """I can't stand it. How can I get rid of it?"""
        pipeline.run(input_text)
    else:
        pipeline.run()

if __name__ == "__main__":
    main()
