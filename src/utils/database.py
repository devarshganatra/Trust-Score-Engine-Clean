"""
Database Integration Module
MongoDB integration for storing trust scores, reviews, and rewards
"""

import pymongo
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import os
import pickle

logger = logging.getLogger(__name__)

class FileStorage:
    """Simple file-based storage fallback when MongoDB is not available"""
    def __init__(self, storage_path: str = "./data/storage"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.reviews_file = os.path.join(storage_path, "reviews.pkl")
        self.trust_scores_file = os.path.join(storage_path, "trust_scores.pkl")
        self.reviewers_file = os.path.join(storage_path, "reviewers.pkl")
        self.rewards_file = os.path.join(storage_path, "rewards.pkl")
        self.reviews = self._load_data(self.reviews_file, {})
        self.trust_scores = self._load_data(self.trust_scores_file, [])
        self.reviewers = self._load_data(self.reviewers_file, {})
        self.rewards = self._load_data(self.rewards_file, [])
    def _load_data(self, filepath: str, default_value):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
        return default_value
    def _save_data(self, filepath: str, data):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Could not save {filepath}: {e}")
            return False
    def save_review(self, review_data: Dict) -> bool:
        try:
            key = f"{review_data.get('reviewerID', '')}_{review_data.get('asin', '')}"
            self.reviews[key] = review_data
            return self._save_data(self.reviews_file, self.reviews)
        except Exception as e:
            logger.error(f"Error saving review: {e}")
            return False
    def save_trust_score(self, trust_score_data: Dict) -> bool:
        try:
            self.trust_scores.append(trust_score_data)
            return self._save_data(self.trust_scores_file, self.trust_scores)
        except Exception as e:
            logger.error(f"Error saving trust score: {e}")
            return False
    def save_reviewer_profile(self, reviewer_data: Dict) -> bool:
        try:
            reviewer_id = reviewer_data.get('reviewerID', '')
            self.reviewers[reviewer_id] = reviewer_data
            return self._save_data(self.reviewers_file, self.reviewers)
        except Exception as e:
            logger.error(f"Error saving reviewer profile: {e}")
            return False
    def save_reward(self, reward_data: Dict) -> bool:
        try:
            self.rewards.append(reward_data)
            return self._save_data(self.rewards_file, self.rewards)
        except Exception as e:
            logger.error(f"Error saving reward: {e}")
            return False
    def get_review(self, reviewer_id: str, product_id: str) -> Optional[Dict]:
        key = f"{reviewer_id}_{product_id}"
        return self.reviews.get(key)
    def get_reviewer_reviews(self, reviewer_id: str, limit: int = 100) -> List[Dict]:
        reviews = []
        for review in self.reviews.values():
            if review.get('reviewerID') == reviewer_id:
                reviews.append(review)
                if len(reviews) >= limit:
                    break
        return reviews
    def get_product_reviews(self, product_id: str, limit: int = 100) -> List[Dict]:
        reviews = []
        for review in self.reviews.values():
            if review.get('asin') == product_id:
                reviews.append(review)
                if len(reviews) >= limit:
                    break
        return reviews
    def get_trust_score(self, review_id: str) -> Optional[Dict]:
        for score in self.trust_scores:
            if score.get('review_id') == review_id:
                return score
        return None
    def get_reviewer_profile(self, reviewer_id: str) -> Optional[Dict]:
        return self.reviewers.get(reviewer_id)
    def get_top_trusted_reviews(self, limit: int = 50) -> List[Dict]:
        sorted_scores = sorted(self.trust_scores, key=lambda x: x.get('trust_score', 0), reverse=True)
        return sorted_scores[:limit]
    def get_suspicious_reviews(self, threshold: float = 30.0, limit: int = 50) -> List[Dict]:
        suspicious = [score for score in self.trust_scores if score.get('trust_score', 100) < threshold]
        sorted_suspicious = sorted(suspicious, key=lambda x: x.get('trust_score', 0))
        return sorted_suspicious[:limit]
    def get_reviewer_rewards(self, reviewer_id: str) -> List[Dict]:
        return [reward for reward in self.rewards if reward.get('reviewerID') == reviewer_id]
    def get_statistics(self) -> Dict:
        stats = {
            'total_reviews': len(self.reviews),
            'total_reviewers': len(self.reviewers),
            'total_trust_scores': len(self.trust_scores),
            'total_rewards': len(self.rewards),
            'avg_trust_score': 0.0,
            'high_trust_reviews': 0,
            'low_trust_reviews': 0
        }
        if self.trust_scores:
            scores = [score.get('trust_score', 0) for score in self.trust_scores]
            stats['avg_trust_score'] = sum(scores) / len(scores)
            stats['high_trust_reviews'] = len([s for s in scores if s >= 70])
            stats['low_trust_reviews'] = len([s for s in scores if s < 30])
        return stats
    def get_all_trust_scores(self):
        return self.trust_scores if hasattr(self, 'trust_scores') else []

class TrustEngineDatabase:
    """MongoDB database interface for Trust Score Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.db = None
        self.collections = {}
        self.use_file_storage = config.get('database', {}).get('use_file_storage', False)
        self.file_storage = None
        
        if self.use_file_storage:
            storage_path = config.get('database', {}).get('file_storage_path', './data/storage')
            self.file_storage = FileStorage(storage_path)
            logger.info(f"Using file storage at: {storage_path}")
            return
        self._connect()
        
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Connect to MongoDB
            self.client = pymongo.MongoClient(self.config['database']['mongodb_uri'])
            self.db = self.client[self.config['database']['database_name']]
            
            # Initialize collections
            collections_config = self.config['database']['collections']
            self.collections = {
                'reviews': self.db[collections_config['reviews']],
                'reviewers': self.db[collections_config['reviewers']],
                'trust_scores': self.db[collections_config['trust_scores']],
                'rewards': self.db[collections_config['rewards']]
            }
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {self.config['database']['database_name']}")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Reviews collection indexes
            self.collections['reviews'].create_index([("reviewerID", pymongo.ASCENDING)])
            self.collections['reviews'].create_index([("asin", pymongo.ASCENDING)])
            self.collections['reviews'].create_index([("unixReviewTime", pymongo.DESCENDING)])
            
            # Trust scores collection indexes
            self.collections['trust_scores'].create_index([("review_id", pymongo.ASCENDING)])
            self.collections['trust_scores'].create_index([("reviewerID", pymongo.ASCENDING)])
            self.collections['trust_scores'].create_index([("trust_score", pymongo.DESCENDING)])
            
            # Reviewers collection indexes
            self.collections['reviewers'].create_index([("reviewerID", pymongo.ASCENDING)])
            
            # Rewards collection indexes
            self.collections['rewards'].create_index([("reviewerID", pymongo.ASCENDING)])
            self.collections['rewards'].create_index([("status", pymongo.ASCENDING)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_review(self, review_data: Dict) -> bool:
        """Save review data to database or file storage"""
        if self.use_file_storage:
            return self.file_storage.save_review(review_data)
        try:
            # Add timestamp
            review_data['created_at'] = datetime.now()
            # Insert or update review
            result = self.collections['reviews'].update_one(
                {'reviewerID': review_data['reviewerID'], 'asin': review_data['asin']},
                {'$set': review_data},
                upsert=True
            )
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error saving review: {e}")
            return False
    
    def save_trust_score(self, trust_score_data: Dict) -> bool:
        """Save trust score to database or file storage"""
        if self.use_file_storage:
            return self.file_storage.save_trust_score(trust_score_data)
        try:
            # Add timestamp
            trust_score_data['created_at'] = datetime.now()
            # Insert trust score
            result = self.collections['trust_scores'].insert_one(trust_score_data)
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error saving trust score: {e}")
            return False
    
    def save_reviewer_profile(self, reviewer_data: Dict) -> bool:
        """Save reviewer profile to database or file storage"""
        if self.use_file_storage:
            return self.file_storage.save_reviewer_profile(reviewer_data)
        try:
            # Add timestamp
            reviewer_data['updated_at'] = datetime.now()
            # Insert or update reviewer profile
            result = self.collections['reviewers'].update_one(
                {'reviewerID': reviewer_data['reviewerID']},
                {'$set': reviewer_data},
                upsert=True
            )
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error saving reviewer profile: {e}")
            return False
    
    def get_review(self, reviewer_id: str, product_id: str) -> Optional[Dict]:
        """Get review by reviewer and product ID from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_review(reviewer_id, product_id)
        try:
            review = self.collections['reviews'].find_one({
                'reviewerID': reviewer_id,
                'asin': product_id
            })
            return review
        except Exception as e:
            logger.error(f"Error getting review: {e}")
            return None
    
    def get_reviewer_reviews(self, reviewer_id: str, limit: int = 100) -> List[Dict]:
        """Get all reviews by a reviewer from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_reviewer_reviews(reviewer_id, limit)
        try:
            reviews = list(self.collections['reviews'].find(
                {'reviewerID': reviewer_id}
            ).sort('unixReviewTime', -1).limit(limit))
            return reviews
        except Exception as e:
            logger.error(f"Error getting reviewer reviews: {e}")
            return []
    
    def get_product_reviews(self, product_id: str, limit: int = 100) -> List[Dict]:
        """Get all reviews for a product from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_product_reviews(product_id, limit)
        try:
            reviews = list(self.collections['reviews'].find(
                {'asin': product_id}
            ).sort('unixReviewTime', -1).limit(limit))
            return reviews
        except Exception as e:
            logger.error(f"Error getting product reviews: {e}")
            return []
    
    def get_trust_score(self, review_id: str) -> Optional[Dict]:
        """Get trust score for a review from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_trust_score(review_id)
        try:
            trust_score = self.collections['trust_scores'].find_one({
                'review_id': review_id
            })
            return trust_score
        except Exception as e:
            logger.error(f"Error getting trust score: {e}")
            return None
    
    def get_reviewer_profile(self, reviewer_id: str) -> Optional[Dict]:
        """Get reviewer profile from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_reviewer_profile(reviewer_id)
        try:
            profile = self.collections['reviewers'].find_one({
                'reviewerID': reviewer_id
            })
            return profile
        except Exception as e:
            logger.error(f"Error getting reviewer profile: {e}")
            return None
    
    def get_top_trusted_reviews(self, limit: int = 50) -> List[Dict]:
        """Get reviews with highest trust scores from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_top_trusted_reviews(limit)
        try:
            reviews = list(self.collections['trust_scores'].find().sort(
                'trust_score', -1
            ).limit(limit))
            return reviews
        except Exception as e:
            logger.error(f"Error getting top trusted reviews: {e}")
            return []
    
    def get_suspicious_reviews(self, threshold: float = 30.0, limit: int = 50) -> List[Dict]:
        """Get reviews with low trust scores (suspicious) from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_suspicious_reviews(threshold, limit)
        try:
            reviews = list(self.collections['trust_scores'].find({
                'trust_score': {'$lt': threshold}
            }).sort('trust_score', 1).limit(limit))
            return reviews
        except Exception as e:
            logger.error(f"Error getting suspicious reviews: {e}")
            return []
    
    def save_reward(self, reward_data: Dict) -> bool:
        """Save reward data to database or file storage"""
        if self.use_file_storage:
            return self.file_storage.save_reward(reward_data)
        try:
            reward_data['created_at'] = datetime.now()
            reward_data['status'] = 'pending'
            result = self.collections['rewards'].insert_one(reward_data)
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error saving reward: {e}")
            return False
    
    def get_reviewer_rewards(self, reviewer_id: str) -> List[Dict]:
        """Get all rewards for a reviewer from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_reviewer_rewards(reviewer_id)
        try:
            rewards = list(self.collections['rewards'].find({
                'reviewerID': reviewer_id
            }).sort('created_at', -1))
            return rewards
        except Exception as e:
            logger.error(f"Error getting reviewer rewards: {e}")
            return []
    
    def update_reward_status(self, reward_id: str, status: str) -> bool:
        """Update reward status"""
        try:
            result = self.collections['rewards'].update_one(
                {'_id': reward_id},
                {'$set': {'status': status, 'updated_at': datetime.now()}}
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error updating reward status: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics from database or file storage"""
        if self.use_file_storage:
            return self.file_storage.get_statistics()
        try:
            stats = {
                'total_reviews': self.collections['reviews'].count_documents({}),
                'total_reviewers': self.collections['reviewers'].count_documents({}),
                'total_trust_scores': self.collections['trust_scores'].count_documents({}),
                'total_rewards': self.collections['rewards'].count_documents({}),
                'avg_trust_score': 0.0,
                'high_trust_reviews': 0,
                'low_trust_reviews': 0
            }
            # Calculate average trust score
            pipeline = [
                {'$group': {'_id': None, 'avg_score': {'$avg': '$trust_score'}}}
            ]
            avg_result = list(self.collections['trust_scores'].aggregate(pipeline))
            if avg_result:
                stats['avg_trust_score'] = round(avg_result[0]['avg_score'], 2)
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        try:
            if self.client:
                self.client.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def get_all_trust_scores(self):
        if self.use_file_storage:
            return self.file_storage.get_all_trust_scores()
        else:
            return list(self.collections['trust_scores'].find({}))


class AsyncTrustEngineDatabase:
    """Async MongoDB interface for Trust Score Engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.db = None
        self.collections = {}
        
    async def connect(self):
        """Connect to MongoDB asynchronously"""
        try:
            self.client = AsyncIOMotorClient(self.config['database']['mongodb_uri'])
            self.db = self.client[self.config['database']['database_name']]
            
            collections_config = self.config['database']['collections']
            self.collections = {
                'reviews': self.db[collections_config['reviews']],
                'reviewers': self.db[collections_config['reviewers']],
                'trust_scores': self.db[collections_config['trust_scores']],
                'rewards': self.db[collections_config['rewards']]
            }
            
            logger.info(f"Async connection to MongoDB established")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise
    
    async def close(self):
        """Close async database connection"""
        try:
            if self.client:
                self.client.close()
                logger.info("Async database connection closed")
        except Exception as e:
            logger.error(f"Error closing async database connection: {e}") 