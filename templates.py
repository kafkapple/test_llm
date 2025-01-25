SUMMARY_TEMPLATE = {
    "korean": """다음 텍스트를 논리적 흐름 순으로 정리 요약해주세요. 중복은 제거하고, 반드시 한국어로 답변하세요.
형식은 마크다운, 2-3 단계 헤딩 및 2-3 단계 - 들여쓰기:

{text}""",
    "english": """Please summarize the following text in logical order, removing duplicates. Answer in English only.
Format in markdown with 2-3 level headings and 2-3 level indentation:

{text}"""
}

EMOTION_TEMPLATE = {
    "korean": """다음 텍스트의 감정을 분석해주세요. 반드시 한국어로 답변하되, 다음 형식의 JSON으로 작성해주세요:

{{
    "emotion": ["기쁨", "분노", "슬픔", "놀람", "혐오", "두려움", "중립" 중 하나],
    "confidence": [0-1 사이의 숫자],
    "reason": [감정 판단 근거],
    "keywords": [감정과 관련된 주요 단어들의 리스트]
}}

입력 텍스트:
{text}""",
    "english": """Please analyze the emotion in the following text. Answer in English only, using this JSON format:

{{
    "emotion": [one of "happy", "angry", "sad", "surprise", "disgust", "fear", "neutral"],
    "confidence": [number between 0-1],
    "reason": [reasoning for emotion],
    "keywords": [list of emotion-related keywords]
}}

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
        "query": QUERY_TEMPLATE
    }
    
    template_dict = templates.get(task_type.lower())
    if template_dict is None:
        return None
        
    return template_dict.get(language.lower(), template_dict["korean"]) 