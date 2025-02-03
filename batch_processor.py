import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from datetime import datetime
from omegaconf import DictConfig
import torch
import re
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np


class BatchProcessor:
    """Process multiple text files or CSV in batches"""
    
    def __init__(self, model, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.output_dir = self._create_output_dir()
        self.batch_size = cfg.batch_processing.get('batch_size', 1)
        
        # Emotion mapping configuration (all lowercase)
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
        """Create directory for saving results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"batch_results_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def process_directory(self, input_dir: Union[str, Path]) -> None:
        """Process all txt files in directory"""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"Directory not found: {input_dir}")
        
        results = []
        true_labels = []
        pred_labels = []
        files = list(input_dir.glob("*.txt"))
        
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            batch_texts = []
            batch_ids = []
            batch_labels = []
            batch_arousal = []
            batch_valence = []
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        batch_texts.append(text)
                        batch_ids.append(file_path.name)
                        batch_labels.append("Unknown")
                        batch_arousal.append(None)
                        batch_valence.append(None)
                except Exception as e:
                    print(f"Error processing file - {file_path}: {str(e)}")
            
            if batch_texts:
                batch_results = self._process_batch(
                    batch_texts, batch_ids, batch_labels,
                    batch_arousal, batch_valence
                )
                results.extend(batch_results)
                
                # Collect labels
                for result in batch_results:
                    if 'emotion' in result:
                        true_labels.append(result['true_label'])
                        pred_labels.append(
                            self._map_emotion(result['emotion'])
                        )
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
        
        self._save_results(results, true_labels, pred_labels)
    
    def process_csv(self, csv_path: Union[str, Path], text_column: str) -> None:
        """Process specific column in CSV file"""
        try:
            df = pd.read_csv(csv_path)
            # 필수 컬럼 검사
            if text_column not in df.columns or 'class' not in df.columns:
                raise ValueError(f"Required columns missing: {text_column} or class")
            
            results = []
            true_labels = []
            pred_labels = []
            
            texts = df[text_column].dropna().tolist()
            labels = df['class'].dropna().tolist()
            
            # arousal과 valence 값이 있으면 가져오고, 없으면 None 리스트 생성
            try:
                arousal_values = df['arousal'].tolist()
            except KeyError:
                print("Arousal column not found. Using None values.")
                arousal_values = [None] * len(texts)
            
            try:
                valence_values = df['valence'].tolist()
            except KeyError:
                print("Valence column not found. Using None values.")
                valence_values = [None] * len(texts)
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                batch_arousal = arousal_values[i:i + self.batch_size]
                batch_valence = valence_values[i:i + self.batch_size]
                batch_ids = [f"row_{i+j}" for j in range(len(batch_texts))]
                
                batch_results = self._process_batch(
                    batch_texts, batch_ids, batch_labels,
                    batch_arousal, batch_valence
                )
                results.extend(batch_results)
                
                # Collect labels
                for result in batch_results:
                    if 'emotion' in result:
                        true_labels.append(result['true_label'])
                        pred_labels.append(
                            self._map_emotion(result['emotion'])
                        )
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
            
            self._save_results(results, true_labels, pred_labels)
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
    
    def _map_emotion(self, emotion: str) -> str:
        """Map emotion label to standardized class"""
        emotion = emotion.lower()
        for standard, variants in self.emotion_mapping.items():
            if emotion in [v.lower() for v in variants]:
                return standard
        return 'unknown'
    
    def _process_batch(
        self,
        texts: List[str],
        source_ids: List[str],
        labels: List[str],
        arousal_values: List[float],
        valence_values: List[float]
    ) -> List[Dict]:
        """Process batch of texts"""
        results = []
        for text, source_id, label, arousal, valence in zip(
            texts, source_ids, labels, arousal_values, valence_values
        ):
            try:
                print(f"\nProcessing: {source_id}")
                response = self.model.generate(text)
                
                # Try JSON parsing
                parsed = None
                if self.cfg.task_type == "emotion":
                    try:
                        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                        matches = re.findall(json_pattern, response, re.DOTALL)
                        if matches:
                            json_str = max(matches, key=len)
                            parsed = json.loads(json_str)
                            
                            required_fields = [
                                "emotion", "confidence", "reason", "keywords",
                                "arousal", "valence"
                            ]
                            if not all(field in parsed for field in required_fields):
                                parsed = None
                    except Exception as e:
                        print(f"JSON parsing failed - {source_id}: {str(e)}")
                        parsed = None
                
                result = {
                    "source_id": source_id,
                    "input_text": text,
                    "true_label": label,
                    "true_arousal": arousal,
                    "true_valence": valence,
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                if parsed:
                    result.update({
                        "emotion": parsed["emotion"],
                        "confidence": parsed["confidence"],
                        "reason": parsed["reason"],
                        "keywords": parsed["keywords"],
                        "arousal": parsed["arousal"],
                        "valence": parsed["valence"]
                    })
                
                results.append(result)
                
            except Exception as e:
                print(f"Processing failed - {source_id}: {str(e)}")
                continue
        
        return results
    
    def _save_results(
        self,
        results: List[Dict],
        true_labels: List[str],
        pred_labels: List[str]
    ) -> None:
        """Save results as JSON, CSV and analysis reports"""
        if not results:
            print("No results to save.")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create comparison table with arousal and valence
        comparison_df = pd.DataFrame({
            'Text': df['input_text'],
            'True Label': df['true_label'],
            'Predicted': df['emotion'],
            'Confidence': df['confidence'],
            'Reason': df['reason']
        })
        
        # arousal과 valence 컬럼이 있는 경우에만 추가
        if 'true_arousal' in df.columns and 'arousal' in df.columns:
            comparison_df['True Arousal'] = df['true_arousal']
            comparison_df['Predicted Arousal'] = df['arousal']
        
        if 'true_valence' in df.columns and 'valence' in df.columns:
            comparison_df['True Valence'] = df['true_valence']
            comparison_df['Predicted Valence'] = df['valence']
        
        # Generate classification report
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        
        # Calculate regression metrics only if values exist
        regression_metrics = {}
        
        try:
            if 'true_arousal' in df.columns and 'arousal' in df.columns:
                # NaN 값 제외하고 계산
                mask = df['true_arousal'].notna() & df['arousal'].notna()
                if mask.any():
                    true_arousal = np.array([float(x) for x in df.loc[mask, 'true_arousal']])
                    pred_arousal = np.array([float(x) for x in df.loc[mask, 'arousal']])
                    regression_metrics['Arousal'] = {
                        'MSE': mean_squared_error(true_arousal, pred_arousal),
                        'RMSE': np.sqrt(mean_squared_error(true_arousal, pred_arousal)),
                        'MAE': mean_absolute_error(true_arousal, pred_arousal),
                        'R2': r2_score(true_arousal, pred_arousal)
                    }
        except Exception as e:
            print(f"Error calculating arousal metrics: {str(e)}")
        
        try:
            if 'true_valence' in df.columns and 'valence' in df.columns:
                # NaN 값 제외하고 계산
                mask = df['true_valence'].notna() & df['valence'].notna()
                if mask.any():
                    true_valence = np.array([float(x) for x in df.loc[mask, 'true_valence']])
                    pred_valence = np.array([float(x) for x in df.loc[mask, 'valence']])
                    regression_metrics['Valence'] = {
                        'MSE': mean_squared_error(true_valence, pred_valence),
                        'RMSE': np.sqrt(mean_squared_error(true_valence, pred_valence)),
                        'MAE': mean_absolute_error(true_valence, pred_valence),
                        'R2': r2_score(true_valence, pred_valence)
                    }
        except Exception as e:
            print(f"Error calculating valence metrics: {str(e)}")
        
        regression_df = pd.DataFrame(regression_metrics).transpose() if regression_metrics else None
        
        # Save files
        json_path = self.output_dir / "results.json"
        csv_path = self.output_dir / "results.csv"
        comparison_path = self.output_dir / "comparison.csv"
        class_report_path = self.output_dir / "classification_report.csv"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        report_df.to_csv(class_report_path, encoding='utf-8')
        
        if regression_df is not None:
            regression_report_path = self.output_dir / "regression_metrics.csv"
            regression_df.to_csv(regression_report_path, encoding='utf-8')
        
        print(f"\n=== Classification Performance Report ===")
        print(classification_report(true_labels, pred_labels))
        
        if regression_metrics:
            print(f"\n=== Regression Metrics Report ===")
            if 'Arousal' in regression_metrics:
                print("\nArousal Metrics:")
                print(f"MSE: {regression_metrics['Arousal']['MSE']:.4f}")
                print(f"RMSE: {regression_metrics['Arousal']['RMSE']:.4f}")
                print(f"MAE: {regression_metrics['Arousal']['MAE']:.4f}")
                print(f"R2: {regression_metrics['Arousal']['R2']:.4f}")
            
            if 'Valence' in regression_metrics:
                print("\nValence Metrics:")
                print(f"MSE: {regression_metrics['Valence']['MSE']:.4f}")
                print(f"RMSE: {regression_metrics['Valence']['RMSE']:.4f}")
                print(f"MAE: {regression_metrics['Valence']['MAE']:.4f}")
                print(f"R2: {regression_metrics['Valence']['R2']:.4f}")
        
        print(f"\nResults have been saved:")
        print(f"JSON: {json_path}")
        print(f"CSV: {csv_path}")
        print(f"Comparison Table: {comparison_path}")
        print(f"Classification Report: {class_report_path}")
        if regression_df is not None:
            print(f"Regression Metrics: {regression_report_path}") 