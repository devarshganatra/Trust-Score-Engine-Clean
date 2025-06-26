"""
Comprehensive Test Suite for Trust Score Engine
Tests all major components and integration points
"""

import pytest
import json
import tempfile
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from pipeline.trust_score_pipeline import TrustScorePipeline
from modules.text_analysis import TextAnalysisPipeline
from modules.reviewer_profiling import ReviewerProfiler
from models.trust_score_model import TrustScoreModel
from utils.evaluation import TrustScoreEvaluator
from utils.database import TrustEngineDatabase


class TestTextAnalysis:
    """Test text analysis pipeline functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'features': {
                'semantic_coherence': {
                    'model_name': 'distilbert-base-nli-mean-tokens',
                    'similarity_threshold': 0.5
                },
                'sentiment_analysis': {
                    'outlier_threshold': 2.0
                },
                'template_detection': {
                    'min_tfidf_score': 0.1,
                    'ngram_range': [2, 4]
                }
            }
        }
    
    @pytest.fixture
    def analyzer(self, config):
        return TextAnalysisPipeline(config)
    
    def test_semantic_coherence_analysis(self, analyzer):
        """Test semantic coherence calculation"""
        review_text = "Great wireless headphones with excellent sound quality"
        product_data = {"title": "Wireless Bluetooth Headphones"}
        
        score = analyzer.analyze_semantic_coherence(review_text, product_data)
        
        assert 0 <= score <= 1
        assert score > 0.3  # Should be reasonably coherent
    
    def test_sentiment_analysis(self, analyzer):
        """Test sentiment analysis"""
        review_text = "This product is absolutely amazing! I love it!"
        
        sentiment = analyzer.calculate_sentiment(review_text)
        
        assert -1 <= sentiment <= 1
        assert sentiment > 0  # Positive sentiment
    
    def test_template_detection(self, analyzer):
        """Test template detection"""
        reviews = [
            {"reviewText": "Great product! Highly recommend!"},
            {"reviewText": "Amazing product! Highly recommend!"},
            {"reviewText": "Excellent product! Highly recommend!"},
            {"reviewText": "This is a completely different review about the product quality."}
        ]
        
        template_score = analyzer.detect_templates(reviews)
        
        assert 0 <= template_score <= 1
        # Should detect some similarity in first 3 reviews
    
    def test_burst_detection(self, analyzer):
        """Test burst detection in reviews"""
        reviews = [
            {"unixReviewTime": int(datetime.now().timestamp())},
            {"unixReviewTime": int((datetime.now() - timedelta(hours=1)).timestamp())},
            {"unixReviewTime": int((datetime.now() - timedelta(hours=2)).timestamp())},
            {"unixReviewTime": int((datetime.now() - timedelta(days=10)).timestamp())}
        ]
        
        burst_score = analyzer.detect_burst_patterns(reviews)
        
        assert 0 <= burst_score <= 1
        # Should detect burst in first 3 reviews


class TestReviewerProfiling:
    """Test reviewer profiling functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'features': {
                'activity_pattern': {
                    'min_reviews': 3,
                    'max_time_gap_days': 365
                },
                'helpfulness': {
                    'min_votes': 5,
                    'helpfulness_threshold': 0.6
                }
            }
        }
    
    @pytest.fixture
    def profiler(self, config):
        return ReviewerProfiler(config)
    
    def test_reviewer_profile_creation(self, profiler):
        """Test creating reviewer profiles"""
        reviewer_id = "R123456"
        reviews = [
            {
                "reviewerID": reviewer_id,
                "asin": "B000123456",
                "overall": 5,
                "verified": True,
                "helpful": [10, 15],
                "unixReviewTime": int(datetime.now().timestamp())
            },
            {
                "reviewerID": reviewer_id,
                "asin": "B000789012",
                "overall": 4,
                "verified": True,
                "helpful": [8, 12],
                "unixReviewTime": int((datetime.now() - timedelta(days=5)).timestamp())
            }
        ]
        
        for review in reviews:
            profiler.add_review(reviewer_id, review)
        
        profile = profiler.get_reviewer_profile(reviewer_id)
        
        assert profile is not None
        assert profile['total_reviews'] == 2
        assert profile['verified_purchase_ratio'] == 1.0
        assert profile['avg_rating'] == 4.5
    
    def test_activity_pattern_analysis(self, profiler):
        """Test activity pattern detection"""
        reviewer_id = "R789012"
        reviews = []
        
        # Create burst pattern (multiple reviews in short time)
        base_time = datetime.now()
        for i in range(5):
            review_time = base_time - timedelta(hours=i)
            reviews.append({
                "reviewerID": reviewer_id,
                "asin": f"B{i:06d}",
                "unixReviewTime": int(review_time.timestamp())
            })
        
        for review in reviews:
            profiler.add_review(reviewer_id, review)
        
        profile = profiler.get_reviewer_profile(reviewer_id)
        
        assert profile['burst_score'] > 0.5  # Should detect burst pattern
    
    def test_helpfulness_analysis(self, profiler):
        """Test helpfulness ratio calculation"""
        reviewer_id = "R345678"
        reviews = [
            {
                "reviewerID": reviewer_id,
                "asin": "B000123456",
                "helpful": [15, 20],  # 75% helpful
                "unixReviewTime": int(datetime.now().timestamp())
            },
            {
                "reviewerID": reviewer_id,
                "asin": "B000789012",
                "helpful": [5, 10],   # 50% helpful
                "unixReviewTime": int((datetime.now() - timedelta(days=1)).timestamp())
            }
        ]
        
        for review in reviews:
            profiler.add_review(reviewer_id, review)
        
        profile = profiler.get_reviewer_profile(reviewer_id)
        
        assert 0.5 <= profile['helpfulness_ratio'] <= 0.8  # Should be around 62.5%


class TestTrustScoreModel:
    """Test trust score model functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'model': {
                'fusion_model': 'logistic_regression',
                'logistic_regression_params': {
                    'random_state': 42,
                    'max_iter': 1000
                },
                'feature_weights': {
                    'semantic_coherence': 0.2,
                    'sentiment_outlier': 0.15,
                    'burstiness': 0.15,
                    'template_score': 0.1,
                    'verified_purchase': 0.1,
                    'helpfulness_ratio': 0.1,
                    'activity_pattern': 0.1,
                    'sentiment_uniformity': 0.1
                }
            },
            'trust_score': {
                'min_score': 0,
                'max_score': 100,
                'normalization_method': 'min_max'
            }
        }
    
    @pytest.fixture
    def model(self, config):
        return TrustScoreModel(config)
    
    def test_feature_vector_creation(self, model):
        """Test feature vector creation"""
        text_features = {
            'semantic_coherence': 0.8,
            'sentiment_outlier_score': 0.1,
            'burst_score': 0.3,
            'template_score': 0.2
        }
        
        reviewer_features = {
            'verified_purchase_ratio': 0.9,
            'helpfulness_ratio': 0.7,
            'activity_pattern_score': 0.6,
            'sentiment_uniformity': 0.8
        }
        
        feature_vector = model.create_feature_vector(text_features, reviewer_features)
        
        assert len(feature_vector) == 8
        assert all(0 <= feature <= 1 for feature in feature_vector)
    
    def test_trust_score_prediction(self, model):
        """Test trust score prediction"""
        text_features = {
            'semantic_coherence': 0.9,
            'sentiment_outlier_score': 0.1,
            'burst_score': 0.2,
            'template_score': 0.1
        }
        
        reviewer_features = {
            'verified_purchase_ratio': 1.0,
            'helpfulness_ratio': 0.8,
            'activity_pattern_score': 0.7,
            'sentiment_uniformity': 0.9
        }
        
        trust_score = model.predict_trust_score(text_features, reviewer_features)
        
        assert 0 <= trust_score <= 100
        assert trust_score > 70  # High-quality features should give high score
    
    def test_model_training(self, model):
        """Test model training with synthetic data"""
        # Generate synthetic training data
        X = np.random.rand(100, 8)  # 100 samples, 8 features
        y = np.random.randint(0, 2, 100)  # Binary labels
        
        success = model.train_model(X, y)
        
        assert success
        assert model.is_trained


class TestTrustScorePipeline:
    """Test complete pipeline functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'database': {
                'use_file_storage': True,
                'file_storage_path': './test_data'
            },
            'processing': {
                'batch_size': 100
            }
        }
    
    @pytest.fixture
    def pipeline(self, config):
        return TrustScorePipeline(config)
    
    def test_data_loading(self, pipeline):
        """Test data loading functionality"""
        # Create temporary test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            reviews = [
                {"reviewerID": "R1", "asin": "B1", "reviewText": "Great product!"},
                {"reviewerID": "R2", "asin": "B1", "reviewText": "Good quality."}
            ]
            for review in reviews:
                f.write(json.dumps(review) + '\n')
            reviews_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            products = [
                {"asin": "B1", "title": "Test Product"}
            ]
            for product in products:
                f.write(json.dumps(product) + '\n')
            products_file = f.name
        
        try:
            success = pipeline.load_data(reviews_file, products_file)
            assert success
            assert len(pipeline.reviews_data) == 2
            assert len(pipeline.products_data) == 1
        finally:
            os.unlink(reviews_file)
            os.unlink(products_file)
    
    def test_pipeline_statistics(self, pipeline):
        """Test pipeline statistics generation"""
        # Mock processed reviews
        pipeline.processed_reviews = [
            {"trust_score": 85},
            {"trust_score": 45},
            {"trust_score": 92}
        ]
        
        stats = pipeline.get_statistics()
        
        assert 'total_reviews' in stats
        assert 'avg_trust_score' in stats
        assert stats['total_reviews'] == 3
        assert 70 <= stats['avg_trust_score'] <= 75  # Should be around 74


class TestEvaluation:
    """Test evaluation module functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'processing': {
                'batch_size': 1000
            }
        }
    
    @pytest.fixture
    def evaluator(self, config):
        return TrustScoreEvaluator(config)
    
    def test_trust_score_validation(self, evaluator):
        """Test trust score validation"""
        trust_scores = [
            {"trust_score": 85},
            {"trust_score": 45},
            {"trust_score": 92},
            {"trust_score": 23},
            {"trust_score": 78}
        ]
        
        validation = evaluator.validate_trust_scores(trust_scores)
        
        assert validation['total_scores'] == 5
        assert validation['mean_score'] == 64.6
        assert validation['min_score'] == 23
        assert validation['max_score'] == 92
        assert 'score_distribution' in validation
    
    def test_baseline_comparison(self, evaluator):
        """Test baseline comparison"""
        trust_scores = [
            {"trust_score": 85, "text_features": {"sentiment_score": 0.8}},
            {"trust_score": 45, "text_features": {"sentiment_score": 0.2}},
            {"trust_score": 92, "text_features": {"sentiment_score": 0.9}}
        ]
        
        comparison = evaluator.compare_with_baselines(trust_scores, ['random', 'sentiment_only'])
        
        assert 'random' in comparison
        assert 'sentiment_only' in comparison
        assert 'mean_score' in comparison['random']
        assert 'mean_score' in comparison['sentiment_only']


class TestIntegration:
    """Integration tests for complete system"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        config = {
            'database': {
                'use_file_storage': True,
                'file_storage_path': './test_data'
            },
            'processing': {
                'batch_size': 10
            }
        }
        
        pipeline = TrustScorePipeline(config)
        
        # Create test data
        reviews = [
            {
                "reviewerID": "R1",
                "asin": "B1",
                "reviewText": "Excellent product with great quality!",
                "overall": 5,
                "verified": True,
                "helpful": [10, 15],
                "unixReviewTime": int(datetime.now().timestamp())
            },
            {
                "reviewerID": "R2",
                "asin": "B1",
                "reviewText": "AMAZING!!! BEST EVER!!! BUY NOW!!!",
                "overall": 5,
                "verified": False,
                "helpful": [2, 5],
                "unixReviewTime": int(datetime.now().timestamp())
            }
        ]
        
        products = [
            {
                "asin": "B1",
                "title": "High-Quality Wireless Headphones",
                "category": "Electronics"
            }
        ]
        
        # Save test data
        os.makedirs('./test_data', exist_ok=True)
        
        with open('./test_data/test_reviews.json', 'w') as f:
            for review in reviews:
                f.write(json.dumps(review) + '\n')
        
        with open('./test_data/test_products.json', 'w') as f:
            for product in products:
                f.write(json.dumps(product) + '\n')
        
        try:
            # Run pipeline
            success = pipeline.load_data('./test_data/test_reviews.json', './test_data/test_products.json')
            assert success
            
            success = pipeline.process_reviews()
            assert success
            
            # Verify results
            assert len(pipeline.processed_reviews) == 2
            assert all('trust_score' in r for r in pipeline.processed_reviews)
            
            # First review should have higher trust score than second
            scores = [r['trust_score'] for r in pipeline.processed_reviews]
            assert scores[0] > scores[1]  # Genuine review > suspicious review
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree('./test_data', ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 