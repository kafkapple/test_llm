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
        print(f"You can exit using these keywords: {', '.join(cfg.task.exit_keywords)}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in cfg.task.exit_keywords:
                print("\n=== Ending chat ===")
                break
            pipeline.process(user_input)
    else:
        # Single task mode (emotion analysis)
        print(f"\n=== Starting {cfg.task_type} analysis ===")
        input_text = """오늘 정신과, 고용센터 상담에서는 같은 주제가 반복되었다. 완벽주의 내려놓기, 그리고 감정을 끄적여서 풀어내기."""
        input_text ="""
물성 다룰때는 Graph NN
Uncertainty 다룰때는 Bayesian deep learning
<Uncertainty quantification using Bayesian NN models>
Overconfident sample / Outlier 걸러낼 수 있음
Two types of uncertainty
Aliatric uncertainty: arise from data inherent noise
Episistemic uncertainty: from model incompleteness
-
데이터가 쌓임에 따라 posterior 에 영향을 준다, 고 이해 가능?
Decision boundary 의 여러 가능성들을
분포 개념 -> 분포와 샘플 거리 계산 (mahalanobids)
신뢰도 추정
—> Bayesain 과 NN 의 결합!
! 가능성: 잘못된 input 에 의한 왜곡 최소화 가능
Mean, variance 모든 corr 이 고려되어 update
>Mc drop out (regul) 을 사용 = gaussian 가정
Cost efficient
Prior 를 bayesian inference 해서 학습 후 inference 에 사용
Inference 할 때도 dropout(mask) 여러번 하면 값 mean var
-> uncertainty measure 가능
-
Bayesian Deep prior
!idea
: VAE 로 data dist 추정 후, bayesian 으로
-> 가우시안 말고 다른 최적 커브 탐색 가능성?
->혹은 가우시안의 조합으로 가능할 수 있게 매핑 및 변환?
data 마다 sparsity 다르므로 regul / normalize 하자!
local feature perturbation 시켜 generalization.
-pretrained word vector 사용
Non-static 사용시,
Semantic similarity
Layer 1개짜리 shallow net
기존엔 word look up table 랜덤이었는데,
Word2vec 가져와 fine tuning
<numpy> logical_andlogical_or -Reductionargmin/max: return index of min, max all: if all trueany: if any truemean, median, std 
CNN
: many copies of the same feature detector for diff position(No need FC) Intractable cosine similarity variational inferenace: 보다 단순한 확률 분포로 근사 P(x) : evidence VAE 는 원점 중심으로, 잘 안변함 centered isotropic? affine transform 이 더 제네럴-> 특수한 경우가 linear transform spherical gaussian simplex: 일반 벡터의 basis mu: globalz: local param -

"""
        pipeline.process(input_text)

if __name__ == "__main__":
    main()
