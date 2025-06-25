"""
Main Trust Score Pipeline
Orchestrates all modules to process reviews and generate trust scores
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.text_analysis import TextAnalysisPipeline
from modules.reviewer_profiling import ReviewerProfiler
from models.trust_score_model import TrustScoreModel
from utils.database import TrustEngineDatabase
from modules.rewards_system import RewardsSystem

logger = logging.getLogger(__name__)


class TrustScorePipeline:
    """Main pipeline for trust score processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.text_analyzer = TextAnalysisPipeline(config)
        self.reviewer_profiler = ReviewerProfiler(config)
        self.trust_model = TrustScoreModel(config)
        self.database = TrustEngineDatabase(config)
        self.rewards_system = RewardsSystem(config, self.database)
        
        # Data storage
        self.reviews_data = []
        self.products_data = {}
        self.processed_reviews = []
        
    def load_data(self, reviews_file: str, products_file: str) -> bool:
        """Load review and product data from JSON files"""
        try:
            logger.info("Loading review data...")
            
            # Load reviews
            reviews = []
            with open(reviews_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        reviews.append(json.loads(line))
            
            self.reviews_data = reviews
            logger.info(f"Loaded {len(reviews)} reviews")
            
            # Load products
            logger.info("Loading product data...")
            products = {}
            with open(products_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        product = json.loads(line)
                        products[product['asin']] = product
            
            self.products_data = products
            logger.info(f"Loaded {len(products)} products")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def process_reviews(self, batch_size: int = 1000) -> bool:
        """Process all reviews through the pipeline"""
        try:
            logger.info("Starting review processing...")
            
            # Process reviews in batches (first pass - feature extraction only)
            for i in tqdm(range(0, len(self.reviews_data), batch_size), desc="Processing reviews"):
                batch = self.reviews_data[i:i + batch_size]
                self._process_batch(batch)
            
            # Finalize text analysis
            self.text_analyzer.finalize_analysis()
            
            # Train model BEFORE scoring reviews
            logger.info("Training model before scoring...")
            if not self.train_model():
                logger.warning("Model training failed, using heuristic scoring")
            
            # Now process with final template scores AND trained model
            logger.info("Re-processing with final template scores and trained model...")
            for i in tqdm(range(0, len(self.reviews_data), batch_size), desc="Final processing"):
                batch = self.reviews_data[i:i + batch_size]
                self._process_batch_final(batch)
            
            logger.info(f"Processed {len(self.processed_reviews)} reviews")
            return True
            
        except Exception as e:
            logger.error(f"Error processing reviews: {e}")
            return False
    
    def _process_batch(self, batch: List[Dict]):
        """Process a batch of reviews"""
        for review in batch:
            try:
                # Get product data
                product_data = self.products_data.get(review.get('asin', ''), {})
                
                # Text analysis
                text_features = self.text_analyzer.analyze_review(review, product_data)
                
                # Add text features to review
                review.update(text_features)
                
                # Add to reviewer profiler
                self.reviewer_profiler.add_review(review.get('reviewerID', ''), review)
                
            except Exception as e:
                logger.error(f"Error processing review {review.get('reviewerID', 'unknown')}: {e}")
    
    def _process_batch_final(self, batch: List[Dict]):
        """Final processing with complete features"""
        for review in batch:
            try:
                # Get product data
                product_data = self.products_data.get(review.get('asin', ''), {})
                
                # Get reviewer profile
                reviewer_id = review.get('reviewerID', '')
                reviewer_profile = self.reviewer_profiler.get_reviewer_profile(reviewer_id)
                
                # Combine features
                text_features = {
                    'semantic_coherence': review.get('semantic_coherence', 0.5),
                    'sentiment_outlier_score': review.get('sentiment_outlier_score', 0.0),
                    'burst_score': review.get('burst_score', 1.0),
                    'template_score': review.get('template_score', 0.5)
                }
                
                # Calculate trust score
                trust_score = self.trust_model.predict_trust_score(text_features, reviewer_profile)
                
                # Create processed review record
                processed_review = {
                    'reviewerID': review.get('reviewerID', ''),
                    'asin': review.get('asin', ''),
                    'reviewText': review.get('reviewText', ''),
                    'overall': review.get('overall', 0),
                    'verified': review.get('verified', False),
                    'helpful': review.get('helpful', [0, 0]),
                    'unixReviewTime': review.get('unixReviewTime', 0),
                    'text_features': text_features,
                    'reviewer_features': reviewer_profile,
                    'trust_score': trust_score,
                    'processed_at': datetime.now()
                }
                
                self.processed_reviews.append(processed_review)
                
                # Save to database
                self.database.save_review(processed_review)
                
                # Save trust score
                trust_score_data = {
                    'review_id': f"{reviewer_id}_{review.get('asin', '')}",
                    'reviewerID': reviewer_id,
                    'asin': review.get('asin', ''),
                    'trust_score': trust_score,
                    'text_features': text_features,
                    'reviewer_features': reviewer_profile,
                    'created_at': datetime.now()
                }
                self.database.save_trust_score(trust_score_data)
                
                # Process rewards
                reward_result = self.rewards_system.process_trust_score_reward(processed_review, trust_score)
                if reward_result['success']:
                    logger.info(f"Reward processed: {reward_result['message']}")
                
            except Exception as e:
                logger.error(f"Error in final processing for review {review.get('reviewerID', 'unknown')}: {e}")
    
    def train_model(self) -> bool:
        """Train the trust score model"""
        try:
            logger.info("Training trust score model...")
            if len(self.processed_reviews) == 0:
                logger.error("No processed reviews available for training")
                return False
            # Prepare training data
            training_data = []
            for review in self.processed_reviews:
                training_record = {
                    'semantic_coherence': review['text_features']['semantic_coherence'],
                    'sentiment_outlier_score': review['text_features']['sentiment_outlier_score'],
                    'burst_score': review['text_features']['burst_score'],
                    'template_score': review['text_features']['template_score'],
                    'verified_purchase_ratio': review['reviewer_features']['verified_purchase_ratio'],
                    'helpfulness_ratio': review['reviewer_features']['helpfulness_ratio'],
                    'activity_pattern_score': review['reviewer_features']['activity_pattern_score'],
                    'sentiment_uniformity_score': review['reviewer_features']['sentiment_uniformity_score'],
                    'verified': review['verified'],
                    'helpfulness_ratio': review['reviewer_features']['helpfulness_ratio'],
                    'template_score': review['text_features']['template_score']
                }
                training_data.append(training_record)
            # Train model
            success = self.trust_model.train(training_data)
            if success:
                logger.info("Model training completed successfully")
                # Save trained model
                model_path = 'models/trained_trust_model.pkl'
                self.trust_model.save_model(model_path)
                # Immediately reload the model to ensure is_trained is set and model is available
                self.trust_model.load_model(model_path)
                return True
            else:
                logger.error("Model training failed")
                return False
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def process_single_review(self, review_data: Dict, product_data: Dict) -> Dict:
        """Process a single review for real-time scoring"""
        try:
            # Text analysis
            text_features = self.text_analyzer.analyze_review(review_data, product_data)
            
            # Get or create reviewer profile
            reviewer_id = review_data.get('reviewerID', '')
            reviewer_profile = self.reviewer_profiler.get_reviewer_profile(reviewer_id)
            
            # Calculate trust score
            trust_score = self.trust_model.predict_trust_score(text_features, reviewer_profile)
            
            # Process rewards
            reward_result = self.rewards_system.process_trust_score_reward(review_data, trust_score)
            
            return {
                'review_id': f"{reviewer_id}_{review_data.get('asin', '')}",
                'trust_score': trust_score,
                'text_features': text_features,
                'reviewer_features': reviewer_profile,
                'reward_result': reward_result,
                'processed_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error processing single review: {e}")
            return {
                'error': str(e),
                'trust_score': 50.0,  # Default neutral score
                'processed_at': datetime.now()
            }
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        try:
            stats = {
                'total_reviews_processed': len(self.processed_reviews),
                'total_products': len(self.products_data),
                'unique_reviewers': len(set(r.get('reviewerID', '') for r in self.processed_reviews)),
                'avg_trust_score': 0.0,
                'high_trust_reviews': 0,
                'low_trust_reviews': 0,
                'rewards_awarded': 0
            }
            
            if self.processed_reviews:
                trust_scores = [r['trust_score'] for r in self.processed_reviews]
                stats['avg_trust_score'] = sum(trust_scores) / len(trust_scores)
                stats['high_trust_reviews'] = sum(1 for score in trust_scores if score >= 70)
                stats['low_trust_reviews'] = sum(1 for score in trust_scores if score < 30)
            
            # Get database statistics
            db_stats = self.database.get_statistics()
            stats.update(db_stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def save_results(self, output_file: str):
        """Save processed results to file"""
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(self.processed_reviews)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.database.close()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main function to run the pipeline"""
    import yaml
    # Load configuration
    with open('config/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Initialize pipeline
    pipeline = TrustScorePipeline(config)
    # Try to load trained model if it exists
    model_path = 'models/trained_trust_model.pkl'
    if os.path.exists(model_path):
        try:
            pipeline.trust_model.load_model(model_path)
            logger.info(f"Loaded trained model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load trained model: {e}")
    try:
        # Load data
        if not pipeline.load_data('data/reviews_Electronics_5.json', 'data/meta_Electronics.json'):
            logger.error("Failed to load data")
            return
        # Process reviews
        if not pipeline.process_reviews():
            logger.error("Failed to process reviews")
            return
        # Save results
        pipeline.save_results('output/processed_reviews.csv')
        # Print statistics
        stats = pipeline.get_statistics()
        logger.info("Pipeline Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main() 