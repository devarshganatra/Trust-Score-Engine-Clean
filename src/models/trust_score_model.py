"""
Trust Score Fusion Model
Combines all features to output a final trust score (0-100)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrustScoreModel:
    """Main trust score fusion model using LogisticRegression"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = [
            'semantic_coherence',
            'sentiment_outlier_score',
            'burst_score',
            'template_score',
            'verified_purchase_ratio',
            'helpfulness_ratio',
            'activity_pattern_score',
            'sentiment_uniformity_score'
        ]
        self.is_trained = False
        
    def prepare_features(self, text_features: Dict, reviewer_features: Dict) -> np.ndarray:
        """Prepare feature vector from text and reviewer analysis"""
        try:
            features = []
            
            # Text analysis features
            features.append(text_features.get('semantic_coherence', 0.5))
            features.append(text_features.get('sentiment_outlier_score', 0.0))
            features.append(text_features.get('burst_score', 1.0))
            features.append(text_features.get('template_score', 0.5))
            
            # Reviewer features
            features.append(reviewer_features.get('verified_purchase_ratio', 0.0))
            features.append(reviewer_features.get('helpfulness_ratio', 0.5))
            features.append(reviewer_features.get('activity_pattern_score', 0.5))
            features.append(reviewer_features.get('sentiment_uniformity_score', 0.5))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def create_training_data(self, reviews_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data from reviews with labels"""
        try:
            features_list = []
            labels = []
            ratings = []
            helpfulness_ratios = []
            review_lengths = []
            sentiment_scores = []
            final_positive_count = 0
            for review in reviews_data:
                text = review.get('reviewText', '')
                review_length = len(text)
                rating = review.get('overall', 3)
                helpful = review.get('helpful', [0, 0])
                helpful_votes = helpful[0] if isinstance(helpful, list) and len(helpful) > 0 else 0
                total_votes = helpful[1] if isinstance(helpful, list) and len(helpful) > 1 else 0
                helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0.0
                sentiment_score = review.get('sentiment_polarity', 0.0)
                features = [
                    review.get('semantic_coherence', 0.5),
                    review.get('sentiment_outlier_score', 0.0),
                    review.get('burst_score', 1.0),
                    review.get('template_score', 0.5),
                    review.get('verified_purchase_ratio', 0.0),
                    review.get('helpfulness_ratio', helpfulness_ratio),
                    review.get('activity_pattern_score', 0.5),
                    review.get('sentiment_uniformity_score', sentiment_score)
                ]
                features_list.append(features)
                ratings.append(rating)
                helpfulness_ratios.append(helpfulness_ratio)
                review_lengths.append(review_length)
                sentiment_scores.append(sentiment_score)
                is_good = (rating >= 4) or (helpfulness_ratio > 0.3) or (review_length > 50) or (sentiment_score > 0)
                label = 1 if is_good else 0
                labels.append(label)
                if is_good:
                    final_positive_count += 1
            # Print feature stats
            print(f"[STATS] Ratings: min={min(ratings)}, max={max(ratings)}, mean={sum(ratings)/len(ratings):.2f}")
            print(f"[STATS] Helpfulness ratio: min={min(helpfulness_ratios):.2f}, max={max(helpfulness_ratios):.2f}, mean={sum(helpfulness_ratios)/len(helpfulness_ratios):.2f}")
            print(f"[STATS] Review length: min={min(review_lengths)}, max={max(review_lengths)}, mean={sum(review_lengths)/len(review_lengths):.2f}")
            print(f"[STATS] Sentiment score: min={min(sentiment_scores):.2f}, max={max(sentiment_scores):.2f}, mean={sum(sentiment_scores)/len(sentiment_scores):.2f}")
            print(f"[DEBUG] Positive labels: {final_positive_count}, Negative labels: {len(labels) - final_positive_count}")
            # Fallback if no positive labels
            if final_positive_count == 0 or final_positive_count == len(labels):
                print("[FALLBACK] Forcing 50/50 label split for model training.")
                half = len(labels) // 2
                labels = [1]*half + [0]*(len(labels)-half)
                final_positive_count = sum(labels)
                print(f"[FALLBACK] New positive labels: {final_positive_count}, Negative labels: {len(labels) - final_positive_count}")
            X = np.array(features_list)
            y = np.array(labels)
            return X, y
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return np.array([]), np.array([])
    
    def train(self, reviews_data: List[Dict], validation_split: float = 0.2):
        """Train the trust score model"""
        try:
            logger.info("Creating training data...")
            X, y = self.create_training_data(reviews_data)
            if len(X) == 0:
                logger.error("No training data available")
                return False
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            # Train model
            self.model = LogisticRegression(
                random_state=42, max_iter=1000, solver="liblinear", class_weight="balanced"
            )
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_val_scaled)
            acc = np.mean(y_pred == y_val)
            logger.info(f"Validation Accuracy: {acc:.3f}")
            logger.info(f"Class balance (train): {np.bincount(y_train)}")
            logger.info(f"Class balance (val): {np.bincount(y_val)}")
            if hasattr(self.model, 'coef_'):
                logger.info(f"Feature importances: {self.model.coef_}")
            # Record decision_function range for calibration (wider spread than probabilities)
            train_scores = self.model.decision_function(X_train_scaled)
            self.score_min = float(train_scores.min())
            self.score_max = float(train_scores.max())
            logger.info(f"Calibrating with decision range min={self.score_min:.4f}, max={self.score_max:.4f}")
            self.is_trained = True
            self.save_model('models/trained_trust_model.pkl')
            logger.info("Model training completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_trust_score(self, text_features: Dict, reviewer_features: Dict) -> float:
        """Predict trust score for a single review"""
        try:
            if not self.is_trained or self.model is None:
                logger.debug("Model not trained. Using default scoring.")
                return self._heuristic_trust_score(text_features, reviewer_features)
            
            # Prepare features
            features = self.prepare_features(text_features, reviewer_features)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction decision score
            raw_score = self.model.decision_function(features_scaled)[0]
            # Calibrate decision score to 0-100 scale
            if hasattr(self, 'score_max') and self.score_max > self.score_min:
                trust_score = ((raw_score - self.score_min) / (self.score_max - self.score_min)) * 100
                trust_score = max(0.0, min(100.0, trust_score))
            else:
                trust_score = float(self.model.predict_proba(features_scaled)[0][1] * 100)
            
            return float(trust_score)
            
        except Exception as e:
            logger.error(f"Error predicting trust score: {e}")
            return self._heuristic_trust_score(text_features, reviewer_features)
    
    def _heuristic_trust_score(self, text_features: Dict, reviewer_features: Dict) -> float:
        """Fallback heuristic scoring when model is not trained"""
        try:
            # Simple weighted combination
            weights = {
                'semantic_coherence': 0.15,
                'sentiment_outlier_score': 0.10,
                'burst_score': 0.10,
                'template_score': 0.15,
                'verified_purchase_ratio': 0.20,
                'helpfulness_ratio': 0.15,
                'activity_pattern_score': 0.10,
                'sentiment_uniformity_score': 0.05
            }
            
            score = 0.0
            
            # Text features
            score += text_features.get('semantic_coherence', 0.5) * weights['semantic_coherence']
            score += (1.0 - min(text_features.get('sentiment_outlier_score', 0.0) / 3.0, 1.0)) * weights['sentiment_outlier_score']
            score += (1.0 - min(text_features.get('burst_score', 1.0) / 5.0, 1.0)) * weights['burst_score']
            score += text_features.get('template_score', 0.5) * weights['template_score']
            
            # Reviewer features
            score += reviewer_features.get('verified_purchase_ratio', 0.0) * weights['verified_purchase_ratio']
            score += reviewer_features.get('helpfulness_ratio', 0.5) * weights['helpfulness_ratio']
            score += reviewer_features.get('activity_pattern_score', 0.5) * weights['activity_pattern_score']
            score += reviewer_features.get('sentiment_uniformity_score', 0.5) * weights['sentiment_uniformity_score']
            
            # Convert to 0-100 scale and clamp
            trust_score = max(0.0, min(100.0, score * 100))
            return float(trust_score)
            
        except Exception as e:
            logger.error(f"Error in heuristic scoring: {e}")
            return 50.0  # Neutral score
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if not self.is_trained or self.model is None:
                return {name: 0.0 for name in self.feature_names}
            
            importance = self.model.coef_[0]
            return dict(zip(self.feature_names, importance))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            if self.is_trained:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'config': self.config,
                    'trained_at': datetime.now().isoformat()
                }
                joblib.dump(model_data, filepath)
                logger.info(f"Model saved to {filepath}")
            else:
                logger.warning("No trained model to save")
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False 