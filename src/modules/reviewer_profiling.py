"""
Reviewer Profiling Module for Trust Score Engine
Analyzes reviewer behavior patterns and credibility indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ReviewerProfiler:
    """Comprehensive reviewer behavior analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reviewer_data = defaultdict(lambda: {
            'reviews': [],
            'sentiments': [],
            'helpful_votes': [],
            'verified_purchases': [],
            'review_times': [],
            'products': set()
        })
        
    def add_review(self, reviewer_id: str, review_data: Dict):
        """Add a review to reviewer's profile"""
        try:
            # Extract review data
            sentiment = review_data.get('sentiment_polarity', 0.0)
            helpful = review_data.get('helpful', [0, 0])  # [helpful_votes, total_votes]
            verified = review_data.get('verified', False)
            review_time = datetime.fromtimestamp(review_data.get('unixReviewTime', 0))
            product_id = review_data.get('asin', '')
            
            # Store in reviewer profile
            self.reviewer_data[reviewer_id]['reviews'].append(review_data)
            self.reviewer_data[reviewer_id]['sentiments'].append(sentiment)
            self.reviewer_data[reviewer_id]['helpful_votes'].append(helpful)  # Store as list
            self.reviewer_data[reviewer_id]['verified_purchases'].append(verified)
            self.reviewer_data[reviewer_id]['review_times'].append(review_time)
            self.reviewer_data[reviewer_id]['products'].add(product_id)
            
        except Exception as e:
            logger.error(f"Error adding review to profile: {e}")
    
    def get_verified_purchase_ratio(self, reviewer_id: str) -> float:
        """Calculate ratio of verified purchases"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.0
                
            verified_purchases = self.reviewer_data[reviewer_id]['verified_purchases']
            if not verified_purchases:
                return 0.0
                
            verified_ratio = sum(verified_purchases) / len(verified_purchases)
            return float(verified_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating verified purchase ratio: {e}")
            return 0.0
    
    def get_helpfulness_ratio(self, reviewer_id: str) -> float:
        """Calculate helpfulness ratio (helpful votes / total votes)"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.5  # Neutral score
                
            helpful_votes = self.reviewer_data[reviewer_id]['helpful_votes']
            if not helpful_votes:
                return 0.5
                
            total_helpful = 0
            total_votes = 0
            
            for helpful_data in helpful_votes:
                # helpful_data is a list [helpful_votes, total_votes]
                if isinstance(helpful_data, list) and len(helpful_data) >= 2:
                    total_helpful += helpful_data[0]  # First element is helpful votes
                    total_votes += helpful_data[1]    # Second element is total votes
                elif isinstance(helpful_data, dict):
                    # Fallback for dict format if it exists
                    total_helpful += helpful_data.get('helpful', 0)
                    total_votes += helpful_data.get('total', 0)
                
            if total_votes == 0:
                return 0.5
                
            helpfulness_ratio = total_helpful / total_votes
            return float(helpfulness_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating helpfulness ratio: {e}")
            return 0.5
    
    def get_activity_pattern_score(self, reviewer_id: str) -> float:
        """Analyze reviewer activity pattern for suspicious behavior"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.5
                
            review_times = self.reviewer_data[reviewer_id]['review_times']
            if len(review_times) < 2:
                return 0.5
                
            # Sort review times
            sorted_times = sorted(review_times)
            
            # Calculate time deltas between consecutive reviews
            time_deltas = []
            for i in range(1, len(sorted_times)):
                delta = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600  # hours
                time_deltas.append(delta)
            
            if not time_deltas:
                return 0.5
                
            # Analyze patterns
            mean_delta = np.mean(time_deltas)
            std_delta = np.std(time_deltas)
            
            # Suspicious patterns:
            # 1. Very regular intervals (low std)
            # 2. Very short intervals (low mean)
            # 3. Very long intervals (high mean)
            
            regularity_score = 1.0 - min(std_delta / (mean_delta + 1), 1.0)
            
            # Normalize mean delta (0-24 hours = normal, >24 = suspicious)
            mean_score = 1.0 - min(mean_delta / 24.0, 1.0)
            
            # Combine scores
            activity_score = (regularity_score + mean_score) / 2.0
            return float(activity_score)
            
        except Exception as e:
            logger.error(f"Error calculating activity pattern score: {e}")
            return 0.5
    
    def get_sentiment_uniformity_score(self, reviewer_id: str) -> float:
        """Calculate sentiment uniformity across reviewer's reviews"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.5
                
            sentiments = self.reviewer_data[reviewer_id]['sentiments']
            if len(sentiments) < 2:
                return 0.5
                
            # Calculate standard deviation of sentiments
            sentiment_std = np.std(sentiments)
            
            # Lower std = more uniform (potentially suspicious)
            # Higher std = more varied (potentially genuine)
            
            # Normalize to 0-1 (0 = very uniform, 1 = very varied)
            uniformity_score = min(sentiment_std, 1.0)
            return float(uniformity_score)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment uniformity: {e}")
            return 0.5
    
    def get_review_count_score(self, reviewer_id: str) -> float:
        """Calculate score based on number of reviews"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.0
                
            review_count = len(self.reviewer_data[reviewer_id]['reviews'])
            
            # Normalize: 1-5 reviews = low score, 6-20 = medium, 20+ = high
            if review_count <= 5:
                return 0.3
            elif review_count <= 20:
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating review count score: {e}")
            return 0.5
    
    def get_product_diversity_score(self, reviewer_id: str) -> float:
        """Calculate score based on product diversity"""
        try:
            if reviewer_id not in self.reviewer_data:
                return 0.0
                
            products = self.reviewer_data[reviewer_id]['products']
            review_count = len(self.reviewer_data[reviewer_id]['reviews'])
            
            if review_count == 0:
                return 0.0
                
            # Calculate diversity ratio
            diversity_ratio = len(products) / review_count
            
            # Higher diversity = better score
            return float(min(diversity_ratio, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating product diversity score: {e}")
            return 0.5
    
    def get_reviewer_profile(self, reviewer_id: str) -> Dict:
        """Get comprehensive reviewer profile with all metrics"""
        try:
            profile = {
                'verified_purchase_ratio': self.get_verified_purchase_ratio(reviewer_id),
                'helpfulness_ratio': self.get_helpfulness_ratio(reviewer_id),
                'activity_pattern_score': self.get_activity_pattern_score(reviewer_id),
                'sentiment_uniformity_score': self.get_sentiment_uniformity_score(reviewer_id),
                'review_count_score': self.get_review_count_score(reviewer_id),
                'product_diversity_score': self.get_product_diversity_score(reviewer_id)
            }
            
            # Calculate composite reviewer score
            weights = {
                'verified_purchase_ratio': 0.25,
                'helpfulness_ratio': 0.20,
                'activity_pattern_score': 0.20,
                'sentiment_uniformity_score': 0.15,
                'review_count_score': 0.10,
                'product_diversity_score': 0.10
            }
            
            composite_score = sum(
                profile[metric] * weights[metric] 
                for metric in weights
            )
            
            profile['composite_reviewer_score'] = float(composite_score)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting reviewer profile: {e}")
            return {
                'verified_purchase_ratio': 0.0,
                'helpfulness_ratio': 0.5,
                'activity_pattern_score': 0.5,
                'sentiment_uniformity_score': 0.5,
                'review_count_score': 0.5,
                'product_diversity_score': 0.5,
                'composite_reviewer_score': 0.5
            }
    
    def get_all_reviewer_profiles(self) -> Dict[str, Dict]:
        """Get profiles for all reviewers"""
        profiles = {}
        for reviewer_id in self.reviewer_data:
            profiles[reviewer_id] = self.get_reviewer_profile(reviewer_id)
        return profiles 