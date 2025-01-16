import pandas as pd
from pipeline import LLMPipeline
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.2", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # 단일 텍스트
    pipeline = LLMPipeline(cfg)
    result = pipeline.process("오늘은 정말 행복한 하루였다!")
    
    # 텍스트 리스트
    texts = [
        "오늘은 정말 행복한 하루였다!",
        "이런 실수를 하다니 너무 화가 난다.",
        "오늘 날씨가 좋네요."
    ]
    results = pipeline.process(texts)
    
    # DataFrame
    df = pd.DataFrame({
        'text': texts,
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    })
    results_df = pipeline.process(df)

if __name__ == "__main__":
    main() 