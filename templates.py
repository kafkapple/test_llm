SUMMARY_TEMPLATE = {
    "korean": """다음 텍스트를 논리적 흐름 순으로 정리 요약해주세요. 중복은 제거하고, 반드시 한국어로 답변하세요.
형식은 마크다운, 2-3 단계 헤딩 및 2-3 단계 - 들여쓰기:

{text}""",
    "english": """Please summarize the following text in logical order, removing duplicates. Answer in English only.
Format in markdown with 2-3 level headings and 2-3 level indentation:

{text}"""
}

EMO_VAL_TEMPLATE = {
    "korean": """Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.
Text: {text}
Intensity Score:
""",
    "english": """Human:
    Task: Evaluate the valence intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (most negative) to 1 (most positive).
Text: {text}
Intensity Score:

Assistant:
"""
}
EMO_AROUSAL_TEMPLATE = {
    "korean": """Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.
Text: {text}
Intensity Score:
""",
    "english": """Human:
    Task: Evaluate the arousal intensity of the writer's mental state based on the text, assigning it a real-valued score from 0 (least arousal) to 1 (most arousal).
Text: {text}
Intensity Score:

Assistant:
"""
}


EMOTION_TEMPLATE = {
    "korean": """다음 텍스트의 감정을 분석해주세요. 
반드시 한국어로 답변하되, 정확히 다음 JSON 형식만 사용해주세요. JSON 외의 다른 텍스트는 포함하지 마세요:

{{
    "emotion": ["기쁨", "분노", "슬픔", "놀람", "혐오", "두려움", "중립" 중 하나],
    "confidence": [0-1 사이의 숫자],
    "arousal": [0-1 사이의 숫자, 감정의 강도/활성화 정도],
    "valence": [0-1 사이의 숫자, 감정의 긍정/부정 정도],
    "reason": [감정 판단 근거를 문장으로 작성],
    "keywords": [감정과 관련된 주요 단어들의 리스트]
}}

참고:
- arousal: 0은 매우 차분함, 1은 매우 격앙됨
- valence: 0은 매우 부정적, 1은 매우 긍정적
- JSON 형식을 정확히 지켜주세요
- JSON 외의 다른 텍스트는 포함하지 마세요

입력 텍스트:
{text}""",
    "english": """Please analyze the emotion in the following text.
Respond ONLY with the following JSON format. Do not include any text outside the JSON:

{{
    "emotion": [one of "happy", "angry", "sad", "surprise", "disgust", "fear", 
               "neutral"],
    "confidence": [number between 0-1],
    "arousal": [number between 0-1, intensity/activation level of emotion],
    "valence": [number between 0-1, positivity/negativity level of emotion],
    "reason": [write a sentence explaining the emotion],
    "keywords": [list of emotion-related keywords]
}}

Note:
- arousal: 0 is very calm, 1 is very excited
- valence: 0 is very negative, 1 is very positive
- Strictly follow the JSON format
- Do not include any text outside the JSON

Input text:
{text}"""
}

QUERY_TEMPLATE = {
    "korean": """{domain}에서 가장 중요한 것과 그 이유를 한국어로 설명해주세요.""",
    "english": """Please explain in English what is most important in {domain} and why."""
}

def get_template(task_type: str, language: str = "korean") -> str:
    """템플릿 및 언어에 따른 지시문 반환"""
    templates = {
        "summary": SUMMARY_TEMPLATE,
        "emotion": EMOTION_TEMPLATE,
        "emotion_intensity": EMO_VAL_TEMPLATE,
        "query": QUERY_TEMPLATE
    }
    
    template_dict = templates.get(task_type.lower())
    if template_dict is None:
        return None
        
    return template_dict.get(language.lower(), template_dict["korean"]) 