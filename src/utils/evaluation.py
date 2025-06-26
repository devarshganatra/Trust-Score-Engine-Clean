"""
Trust Score Engine Evaluation Module
Provides comprehensive evaluation metrics, validation, and benchmarking
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import time
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class TrustScoreEvaluator:
    """Comprehensive evaluator for trust score engine performance"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.benchmarks = {}
        
    def evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series, 
                                 model, test_size: float = 0.2) -> Dict:
        """Evaluate model performance with comprehensive metrics"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train and predict
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        training_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_time_seconds': training_time,
            'test_samples': len(y_test),
            'train_samples': len(y_train)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        self.results['model_performance'] = metrics
        return metrics
    
    def benchmark_processing_speed(self, reviews_data: List[Dict], 
                                 pipeline) -> Dict:
        """Benchmark processing speed and memory usage"""
        
        import psutil
        import gc
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process reviews
        start_time = time.time()
        
        # Process in batches
        batch_size = self.config.get('processing', {}).get('batch_size', 1000)
        total_reviews = len(reviews_data)
        
        processed_count = 0
        for i in range(0, total_reviews, batch_size):
            batch = reviews_data[i:i + batch_size]
            # Simulate processing (replace with actual pipeline call)
            processed_count += len(batch)
        
        processing_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Calculate throughput
        reviews_per_second = total_reviews / processing_time
        memory_per_review = memory_used / total_reviews
        
        benchmarks = {
            'total_reviews': total_reviews,
            'processing_time_seconds': processing_time,
            'reviews_per_second': reviews_per_second,
            'memory_used_mb': memory_used,
            'memory_per_review_mb': memory_per_review,
            'batch_size': batch_size
        }
        
        self.benchmarks['processing_speed'] = benchmarks
        return benchmarks
    
    def validate_trust_scores(self, trust_scores: List[Dict]) -> Dict:
        """Validate trust score distribution and quality"""
        
        scores = [ts['trust_score'] for ts in trust_scores if 'trust_score' in ts]
        
        if not scores:
            return {'error': 'No trust scores found'}
        
        scores = np.array(scores)
        
        validation = {
            'total_scores': len(scores),
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'score_distribution': {
                'very_low_0_20': int(np.sum(scores <= 20)),
                'low_21_40': int(np.sum((scores > 20) & (scores <= 40))),
                'medium_41_60': int(np.sum((scores > 40) & (scores <= 60))),
                'high_61_80': int(np.sum((scores > 60) & (scores <= 80))),
                'very_high_81_100': int(np.sum(scores > 80))
            },
            'outliers': {
                'iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                'outlier_count': int(np.sum((scores < np.percentile(scores, 25) - 1.5 * (np.percentile(scores, 75) - np.percentile(scores, 25))) | 
                                           (scores > np.percentile(scores, 75) + 1.5 * (np.percentile(scores, 75) - np.percentile(scores, 25)))))
            }
        }
        
        self.results['trust_score_validation'] = validation
        return validation
    
    def compare_with_baselines(self, trust_scores: List[Dict], 
                             baseline_methods: List[str]) -> Dict:
        """Compare trust scores with baseline methods"""
        
        comparison = {}
        
        for method in baseline_methods:
            if method == 'random':
                # Random baseline
                random_scores = np.random.uniform(0, 100, len(trust_scores))
                comparison['random'] = {
                    'mean_score': float(np.mean(random_scores)),
                    'std_score': float(np.std(random_scores))
                }
            
            elif method == 'sentiment_only':
                # Sentiment-based baseline
                sentiment_scores = []
                for ts in trust_scores:
                    if 'text_features' in ts and 'sentiment_score' in ts['text_features']:
                        sentiment = ts['text_features']['sentiment_score']
                        # Convert sentiment to trust score (0-100)
                        trust_score = (sentiment + 1) * 50  # Assuming sentiment is -1 to 1
                        sentiment_scores.append(trust_score)
                
                if sentiment_scores:
                    comparison['sentiment_only'] = {
                        'mean_score': float(np.mean(sentiment_scores)),
                        'std_score': float(np.std(sentiment_scores)),
                        'correlation_with_trust': float(np.corrcoef(sentiment_scores, 
                                                                  [ts['trust_score'] for ts in trust_scores[:len(sentiment_scores)]])[0, 1])
                    }
            
            elif method == 'helpfulness_only':
                # Helpfulness-based baseline
                helpfulness_scores = []
                for ts in trust_scores:
                    if 'reviewer_features' in ts and 'helpfulness_ratio' in ts['reviewer_features']:
                        helpfulness = ts['reviewer_features']['helpfulness_ratio']
                        trust_score = helpfulness * 100
                        helpfulness_scores.append(trust_score)
                
                if helpfulness_scores:
                    comparison['helpfulness_only'] = {
                        'mean_score': float(np.mean(helpfulness_scores)),
                        'std_score': float(np.std(helpfulness_scores)),
                        'correlation_with_trust': float(np.corrcoef(helpfulness_scores, 
                                                                  [ts['trust_score'] for ts in trust_scores[:len(helpfulness_scores)]])[0, 1])
                    }
        
        self.results['baseline_comparison'] = comparison
        return comparison
    
    def generate_performance_report(self, output_file: str = None) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("# Trust Score Engine Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance
        if 'model_performance' in self.results:
            mp = self.results['model_performance']
            report.append("## Model Performance")
            report.append(f"- Accuracy: {mp['accuracy']:.3f}")
            report.append(f"- Precision: {mp['precision']:.3f}")
            report.append(f"- Recall: {mp['recall']:.3f}")
            report.append(f"- F1 Score: {mp['f1_score']:.3f}")
            if 'roc_auc' in mp:
                report.append(f"- ROC AUC: {mp['roc_auc']:.3f}")
            report.append(f"- Training Time: {mp['training_time_seconds']:.2f} seconds")
            report.append(f"- Cross-validation F1: {mp['cv_f1_mean']:.3f} ± {mp['cv_f1_std']:.3f}")
            report.append("")
        
        # Processing Benchmarks
        if 'processing_speed' in self.benchmarks:
            ps = self.benchmarks['processing_speed']
            report.append("## Processing Performance")
            report.append(f"- Total Reviews: {ps['total_reviews']:,}")
            report.append(f"- Processing Time: {ps['processing_time_seconds']:.2f} seconds")
            report.append(f"- Throughput: {ps['reviews_per_second']:.1f} reviews/second")
            report.append(f"- Memory Usage: {ps['memory_used_mb']:.1f} MB")
            report.append(f"- Memory per Review: {ps['memory_per_review_mb']:.3f} MB")
            report.append("")
        
        # Trust Score Validation
        if 'trust_score_validation' in self.results:
            tv = self.results['trust_score_validation']
            report.append("## Trust Score Validation")
            report.append(f"- Total Scores: {tv['total_scores']:,}")
            report.append(f"- Mean Score: {tv['mean_score']:.2f}")
            report.append(f"- Median Score: {tv['median_score']:.2f}")
            report.append(f"- Standard Deviation: {tv['std_score']:.2f}")
            report.append(f"- Score Range: {tv['min_score']:.1f} - {tv['max_score']:.1f}")
            report.append("")
            report.append("### Score Distribution:")
            for range_name, count in tv['score_distribution'].items():
                percentage = (count / tv['total_scores']) * 100
                report.append(f"- {range_name}: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        # Baseline Comparison
        if 'baseline_comparison' in self.results:
            report.append("## Baseline Comparison")
            for method, metrics in self.results['baseline_comparison'].items():
                report.append(f"### {method.replace('_', ' ').title()}")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        report.append(f"- {key.replace('_', ' ').title()}: {value}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_results(self, output_dir: str = "output"):
        """Save all evaluation results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save benchmarks
        with open(f"{output_dir}/benchmarks.json", 'w') as f:
            json.dump(self.benchmarks, f, indent=2, default=str)
        
        # Generate and save report
        report = self.generate_performance_report(f"{output_dir}/performance_report.md")
        
        logger.info(f"Evaluation results saved to {output_dir}/")
        return report 