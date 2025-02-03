# 필요한 라이브러리 재로딩
import pandas as pd
import numpy as np

# 일반적인 감정 인식 레이블 정의
standard_emotions = {
    "neutral": [
        "I feel okay today.",
        "Nothing special is happening.",
        "Just another ordinary day.",
        "I am neither happy nor sad.",
        "Everything is fine, I guess.",
        "I don’t feel much emotion right now.",
        "Nothing is particularly exciting or upsetting.",
        "I am just going through my routine.",
        "A regular, uneventful day.",
        "Everything is as expected."
    ],
    "happy": [
        "I am feeling absolutely fantastic!",
        "Today is such a beautiful day.",
        "I can't stop smiling, everything is perfect.",
        "This is the happiest moment of my life!",
        "I feel so grateful and excited.",
        "Happiness is all around me today.",
        "I am full of positive energy!",
        "This news just made my day.",
        "Every moment is filled with joy!",
        "Nothing can bring me down today."
    ],
    "sad": [
        "I feel so alone and lost.",
        "This is one of the worst days ever.",
        "I can't seem to find happiness anymore.",
        "Everything feels so empty right now.",
        "My heart is heavy with sorrow.",
        "I just want to be left alone.",
        "Tears won't stop falling from my eyes.",
        "I feel like nothing matters anymore.",
        "Sadness is overwhelming me.",
        "I don't know how to move forward."
    ],
    "angry": [
        "I am absolutely furious right now!",
        "This is so frustrating, I can't believe it!",
        "How could they do this to me?",
        "I can't stand this situation anymore.",
        "Everything is making me so angry!",
        "I feel like yelling at someone.",
        "This is totally unacceptable!",
        "I'm losing my patience completely.",
        "This is driving me insane!",
        "I need to cool down before I explode."
    ],
    "fear": [
        "That was the scariest moment of my life!",
        "I am trembling with fear.",
        "I feel so anxious and uneasy.",
        "My heart is pounding so fast.",
        "I can't stop thinking about what might happen.",
        "This situation makes me feel so helpless.",
        "I just want to run away from this.",
        "I don't feel safe at all.",
        "Something feels terribly wrong.",
        "I am completely terrified right now."
    ],
    "surprise": [
        "I can't believe what just happened!",
        "This is the most unexpected thing ever!",
        "Wow! I was not prepared for this at all.",
        "I am absolutely shocked!",
        "This was a complete surprise!",
        "I never saw this coming!",
        "My mind is blown right now!",
        "This is way beyond my expectations!",
        "I am in total disbelief!",
        "This caught me completely off guard!"
    ],
    "disgust": [
        "That was absolutely disgusting!",
        "I feel so grossed out right now.",
        "I can't stand the sight of this.",
        "This makes me feel so sick.",
        "I feel completely repulsed.",
        "This is beyond nasty!",
        "I have never felt this disgusted before.",
        "This is so unbearable to watch.",
        "I can't handle this awful smell!",
        "Everything about this is just revolting."
    ]
}

# 새로운 감정 레이블에 맞춰 샘플 데이터 생성
data_standard = []
np.random.seed(42)

for category, texts in standard_emotions.items():
    for i, text in enumerate(texts):
        if category == "happy":
            arousal = round(np.random.uniform(0.6, 0.9), 2)
            valence = round(np.random.uniform(0.7, 0.9), 2)
        elif category == "sad":
            arousal = round(np.random.uniform(0.1, 0.4), 2)
            valence = round(np.random.uniform(0.1, 0.4), 2)
        elif category == "angry":
            arousal = round(np.random.uniform(0.7, 0.9), 2)
            valence = round(np.random.uniform(0.1, 0.3), 2)
        elif category == "fear":
            arousal = round(np.random.uniform(0.7, 0.9), 2)
            valence = round(np.random.uniform(0.1, 0.4), 2)
        elif category == "surprise":
            arousal = round(np.random.uniform(0.6, 0.9), 2)
            valence = round(np.random.uniform(0.5, 0.8), 2)
        elif category == "disgust":
            arousal = round(np.random.uniform(0.3, 0.7), 2)
            valence = round(np.random.uniform(0.1, 0.3), 2)
        elif category == "neutral":
            arousal = round(np.random.uniform(0.3, 0.5), 2)
            valence = round(np.random.uniform(0.4, 0.6), 2)
        
        data_standard.append([text, category, arousal, valence])

# 데이터프레임 생성
df_standard = pd.DataFrame(data_standard, columns=["text", "categorical_emotion", "arousal", "valence"])

# 감정별 arousal-valence 값의 분포 확인
emotion_mapping_check = df_standard.groupby("categorical_emotion")[["arousal", "valence"]].agg(['min', 'max', 'mean', 'std'])

# 감정별 샘플 데이터 일부 확인
sample_check = df_standard.groupby("categorical_emotion").head(3)  # 각 감정별 3개 샘플 확인

df_standard.to_csv("emotion_text_data_standard.csv", index=False)

# Happy: 높은 Valence (0.70.9), 높은 Arousal (0.60.9)
# Sad: 낮은 Valence (0.10.4), 낮은 Arousal (0.10.4)
# Angry: 낮은 Valence (0.10.3), 높은 Arousal (0.70.9)
# Fear: 낮은 Valence (0.10.4), 높은 Arousal (0.70.9)
# Surprise: 중간-높은 Valence (0.50.8), 높은 Arousal (0.60.9)
# Disgust: 낮은 Valence (0.10.3), 중간 Arousal (0.30.7)
# Neutral: 중간 Valence (0.40.6), 중간 Arousal (0.30.5)

import matplotlib.pyplot as plt

# 감정별 색상 지정
emotion_colors = {
    "neutral": "gray",
    "happy": "gold",
    "sad": "blue",
    "angry": "red",
    "fear": "purple",
    "surprise": "orange",
    "disgust": "green"
}

# 플롯 생성
plt.figure(figsize=(8, 6))

# 감정별로 점을 찍음
for emotion in df_standard["categorical_emotion"].unique():
    subset = df_standard[df_standard["categorical_emotion"] == emotion]
    plt.scatter(subset["valence"], subset["arousal"], 
                label=emotion, 
                color=emotion_colors[emotion], alpha=0.7)

# 축 레이블 및 제목 설정
plt.xlabel("Valence (Unpleasant → Pleasant)")
plt.ylabel("Arousal (Calm → Excited)")
plt.title("Emotion Mapping on Arousal-Valence Space")
plt.axhline(0.5, color='black', linestyle='dashed', alpha=0.5)  # Arousal 중간선
plt.axvline(0.5, color='black', linestyle='dashed', alpha=0.5)  # Valence 중간선
plt.legend()
plt.grid(True)

# 그래프 출력
plt.show()
