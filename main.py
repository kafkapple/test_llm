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
        input_text = """moser lab 의 grid cell 관련 연구, x y 축 반복되는 패턴 toroidal 

어릴 때 눈 못뜰 때 recording, task 못하니
grid cell 은 없는데, toroidal manifold 는 존재

x y 축 2d space 에서 돌아다니니
collection

task 발생하기 전부터 manifold 발생한다. 
mean? attractor 도 비슷하다 생각. 
task dependent 하지 않은 그런 attractor 좀 더 멋지지 않나. 


ersistent activity 를 유지하는 메커니즘으로 최근 Andermann 랩에서 주장하는 cAMP accumulation 이 여기서도 마찬가지로 적용될 수도 있을 것 같다는 생각이드네요. 저번에 지수님 sweet adaptation 도 cAMP accumulation 관련 메커니즘이었지 않았나요.

camp 같은 2nd messenger 역할 중요할것으로 보임?"""
        input_text = """근데, 아직 1년도 안됐어. 그리고 난 아직 제대로 뭔가 풀어낸 적도 없다고. 억울하고 화가나는데 젠장."""
        pipeline.process(input_text)

if __name__ == "__main__":
    main()
