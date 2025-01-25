import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
from omegaconf import DictConfig
import torch
import re


class BatchProcessor:
    """여러 텍스트 파일 또는 CSV를 배치로 처리하는 클래스"""
    
    def __init__(self, model, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.output_dir = self._create_output_dir()
        self.batch_size = cfg.batch_processing.get('batch_size', 1)
    
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
        files = list(input_dir.glob("*.txt"))
        
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            batch_texts = []
            batch_ids = []
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        batch_texts.append(text)
                        batch_ids.append(file_path.name)
                except Exception as e:
                    print(f"파일 처리 중 오류 발생 - {file_path}: {str(e)}")
            
            if batch_texts:
                batch_results = self._process_batch(batch_texts, batch_ids)
                results.extend(batch_results)
                
            # CUDA 캐시 정리
            torch.cuda.empty_cache()
        
        self._save_results(results)
    
    def process_csv(self, csv_path: Union[str, Path], text_column: str) -> None:
        """CSV 파일의 특정 컬럼 처리"""
        try:
            df = pd.read_csv(csv_path)
            if text_column not in df.columns:
                raise ValueError(f"컬럼을 찾을 수 없습니다: {text_column}")
            
            results = []
            texts = df[text_column].dropna().tolist()
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_ids = [f"row_{i+j}" for j in range(len(batch_texts))]
                
                batch_results = self._process_batch(batch_texts, batch_ids)
                results.extend(batch_results)
                
                # CUDA 캐시 정리
                torch.cuda.empty_cache()
            
            self._save_results(results)
            
        except Exception as e:
            print(f"CSV 처리 중 오류 발생: {str(e)}")
    
    def _process_batch(
        self,
        texts: List[str],
        source_ids: List[str]
    ) -> List[Dict]:
        """텍스트 배치 처리"""
        results = []
        for text, source_id in zip(texts, source_ids):
            try:
                print(f"\n처리 중: {source_id}")
                response = self.model.generate(text)
                
                # JSON 파싱 시도
                parsed = None
                if self.cfg.task_type == "emotion":
                    try:
                        json_pattern = r'\{[^{}]*\}'
                        matches = re.findall(json_pattern, response)
                        if matches:
                            json_str = max(matches, key=len)
                            parsed = json.loads(json_str)
                            
                            # 필수 필드 검증
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
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 파싱된 결과가 있으면 추가
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
    
    def _save_results(self, results: List[Dict]) -> None:
        """결과를 JSON 및 CSV 형식으로 저장"""
        if not results:
            print("저장할 결과가 없습니다.")
            return
        
        # JSON 저장
        json_path = self.output_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV 저장
        csv_path = self.output_dir / "results.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"결과가 저장되었습니다:\nJSON: {json_path}\nCSV: {csv_path}") 