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
        input_text = """topology manifold
bijection: 1:1 대응

그저 점들의 집합일 뿐, 구분 불가능하나,
topology 로 구분 가
연결 구조 neighborhood structure

homeomorphism
connectedness 등 property
Manifold
homology
구멍 수 셈으로써 분류 가능하다는 이야기
동일한 종류의 manifold 
connectedness / 구멍 

국소적 locally manifold = Euclidean (우리가 친숙한)
마치 지구가 둥글다고 하나, 우리에게는 평평해 보이는 locally 

2차원 위치 좌표 

집합과 공간을 구분하는 것은?
나라는 원소 주변은?

homeomorphism
일종의 kernel 로 볼수도 있

거리 개념은 날아가고, 굉장히 

구멍 갯수만 보존

A unified theory for the computational and mechanistic origins of grid cells
저도 이 논문 보면서 topology에 대한 배경지식의 필요성을 느꼈어요"""
        #input_text = """근데, 아직 1년도 안됐어. 그리고 난 아직 제대로 뭔가 풀어낸 적도 없다고. 억울하고 화가나는데 젠장."""
        pipeline.process(input_text)

if __name__ == "__main__":
    main()
