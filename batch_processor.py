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
            
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        batch_texts.append(text)
                        batch_ids.append(file_path.name)
                        batch_labels.append("Unknown")
                except Exception as e:
                    print(f"Error processing file - {file_path}: {str(e)}")
            
            if batch_texts:
                batch_results = self._process_batch(
                    batch_texts, batch_ids, batch_labels
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
            if text_column not in df.columns or 'class' not in df.columns:
                raise ValueError(f"Required columns missing: {text_column} or class")
            
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
        labels: List[str]
    ) -> List[Dict]:
        """Process batch of texts"""
        results = []
        for text, source_id, label in zip(texts, source_ids, labels):
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
                                "emotion", "confidence", "reason", "keywords"
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
                print(f"Processing failed - {source_id}: {str(e)}")
                continue
        
        return results
    
    def _save_results(
        self,
        results: List[Dict],
        true_labels: List[str],
        pred_labels: List[str]
    ) -> None:
        """Save results as JSON, CSV and analysis report"""
        if not results:
            print("No results to save.")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Text': df['input_text'],
            'True Label': df['true_label'],
            'Predicted': df['emotion'],
            'Confidence': df['confidence'],
            'Reason': df['reason']
        })
        
        # Generate classification report
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        
        # Save files
        json_path = self.output_dir / "results.json"
        csv_path = self.output_dir / "results.csv"
        comparison_path = self.output_dir / "comparison.csv"
        report_path = self.output_dir / "classification_report.csv"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
        report_df.to_csv(report_path, encoding='utf-8')
        
        print(f"\n=== Classification Performance Report ===")
        print(classification_report(true_labels, pred_labels))
        print(f"\nResults have been saved:")
        print(f"JSON: {json_path}")
        print(f"CSV: {csv_path}")
        print(f"Comparison Table: {comparison_path}")
        print(f"Classification Report: {report_path}") 