import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
from omegaconf import DictConfig
import torch
import re
from sklearn.metrics import classification_report


class BatchProcessor:
    """여러 텍스트 파일 또는 CSV를 배치로 처리하는 클래스"""
    
    def __init__(self, model, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.output_dir = self._create_output_dir()
        self.batch_size = cfg.batch_processing.get('batch_size', 1)
        
        # 감정 매핑 설정 (모두 소문자로 통일)
        self.emotion_mapping = {
            'happy': ['happy', '기쁨'],
            'sad': ['sad', '슬픔'],
            'angry': ['angry', '분노'],
            'fear': ['fear', '두려움'],
            'surprise': ['surprise', '놀람'],
            'disgust': ['disgust', '혐오'],
            'neutral': ['neutral', '중립']
        }
    
    def _create_output_dir(self) -> Path:
        """결과 저장을 위한 디렉토리 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"batch_results_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def process_directory(self, input_dir: Union[str, Path]) -> None:
        """디렉토리 내의 모든 txt 파일 처리"""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"디렉토리를 찾을 수 없습니다: {input_dir}")
        
        results = []
        true_labels = []
        pred_labels = []
        files = list(input_dir.glob("*.txt"))
        
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            batch_texts = []
            batch_ids = []
            batch_labels = []
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        batch_texts.append(text)
                        batch_ids.append(file_path.name)
                        batch_labels.append("Unknown")
                except Exception as e:
                    print(f"파일 처리 중 오류 발생 - {file_path}: {str(e)}")
            
            if batch_texts:
                batch_results = self._process_batch(
                    batch_texts, batch_ids, batch_labels
                )
                results.extend(batch_results)
                
                # 레이블 수집
                for result in batch_results:
                    if 'emotion' in result:
                        true_labels.append(result['true_label'])
                        pred_labels.append(
                            self._map_emotion(result['emotion'])
                        )
            
            # CUDA 캐시 정리
            torch.cuda.empty_cache()
        
        self._save_results(results, true_labels, pred_labels)
    
    def process_csv(self, csv_path: Union[str, Path], text_column: str) -> None:
        """CSV 파일의 특정 컬럼 처리"""
        try:
            df = pd.read_csv(csv_path)
            if text_column not in df.columns or 'class' not in df.columns:
                raise ValueError(f"필요한 컬럼이 없습니다: {text_column} 또는 class")
            
            results = []
            true_labels = []
            pred_labels = []
            
            texts = df[text_column].dropna().tolist()
            labels = df['class'].dropna().tolist()
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                batch_ids = [f"row_{i+j}" for j in range(len(batch_texts))]
                
                batch_results = self._process_batch(
                    batch_texts, batch_ids, batch_labels
                )
                results.extend(batch_results)
                
                # 레이블 수집
                for result in batch_results:
                    if 'emotion' in result:
                        true_labels.append(result['true_label'])
                        pred_labels.append(
                            self._map_emotion(result['emotion'])
                        )
                
                # CUDA 캐시 정리
                torch.cuda.empty_cache()
            
            self._save_results(results, true_labels, pred_labels)
            
        except Exception as e:
            print(f"CSV 처리 중 오류 발생: {str(e)}")
    
    def _map_emotion(self, emotion: str) -> str:
        """감정 레이블을 표준화된 클래스로 매핑"""
        emotion = emotion.lower()
        for standard, variants in self.emotion_mapping.items():
            if emotion in [v.lower() for v in variants]:
                return standard
        return 'unknown'  # 소문자로 변경
    
    def _process_batch(
        self,
        texts: List[str],
        source_ids: List[str],
        labels: List[str]
    ) -> List[Dict]:
        """텍스트 배치 처리"""
        results = []
        for text, source_id, label in zip(texts, source_ids, labels):
            try:
                print(f"\n처리 중: {source_id}")
                response = self.model.generate(text)
                
                # JSON 파싱 시도
                parsed = None
                if self.cfg.task_type == "emotion":
                    try:
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        matches = re.findall(json_pattern, response, re.DOTALL)
                        if matches:
                            json_str = max(matches, key=len)
                            parsed = json.loads(json_str)
                            
                            required_fields = [
                                "emotion", "confidence", "reason", "keywords"
                            ]
                            if not all(field in parsed for field in required_fields):
                                parsed = None
                    except Exception as e:
                        print(f"JSON 파싱 실패 - {source_id}: {str(e)}")
                        parsed = None
                
                result = {
                    "source_id": source_id,
                    "input_text": text,
                    "true_label": label,
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                if parsed:
                    result.update({
                        "emotion": parsed["emotion"],
                        "confidence": parsed["confidence"],
                        "reason": parsed["reason"],
                        "keywords": parsed["keywords"]
                    })
                
                results.append(result)
                
            except Exception as e:
                print(f"처리 실패 - {source_id}: {str(e)}")
                continue
        
        return results
    
    def _save_results(
        self,
        results: List[Dict],
        true_labels: List[str],
        pred_labels: List[str]
    ) -> None:
        """결과를 JSON, CSV 및 분석 리포트로 저장"""
        if not results:
            print("저장할 결과가 없습니다.")
            return
        
        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(results)
        
        # 비교 테이블 생성
        comparison_df = pd.DataFrame({
            'Text': df['input_text'],
            'True Label': df['true_label'],
            'Predicted': df['emotion'],
            'Confidence': df['confidence'],
            'Reason': df['reason']
        })
        
        # 분류 리포트 생성
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        
        # 파일 저장
        json_path = self.output_dir / "results.json"
        csv_path = self.output_dir / "results.csv"
        comparison_path = self.output_dir / "comparison.csv"
        report_path = self.output_dir / "classification_report.csv"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        report_df.to_csv(report_path, encoding='utf-8')
        
        print(f"\n=== 분류 성능 보고서 ===")
        print(classification_report(true_labels, pred_labels))
        print(f"\n결과가 저장되었습니다:")
        print(f"JSON: {json_path}")
        print(f"CSV: {csv_path}")
        print(f"비교 테이블: {comparison_path}")
        print(f"분류 리포트: {report_path}") 