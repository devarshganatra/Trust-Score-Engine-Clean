"""
Text Analysis Modules for Trust Score Engine
Implements semantic coherence, sentiment analysis, burstiness, and template detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SemanticCoherenceAnalyzer:
    """Analyzes semantic coherence between review text and product description"""
    
    def __init__(self, model_name: str = "distilbert-base-nli-mean-tokens"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode([text])[0]
        self.cache[text] = embedding
        return embedding
    
    def calculate_coherence(self, review_text: str, product_description: str) -> float:
        """Calculate semantic coherence score between review and product"""
        try:
            if not review_text or not product_description:
                return 0.5  # Neutral score for missing data
                
            review_embedding = self.get_embedding(review_text)
            product_embedding = self.get_embedding(product_description)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [review_embedding], [product_embedding]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.5


class SentimentAnalyzer:
    """Analyzes sentiment and detects outliers"""
    
    def __init__(self):
        self.product_sentiments = {}
        self.category_sentiments = {}
        
    def get_sentiment_polarity(self, text: str) -> float:
        """Get sentiment polarity using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def calculate_outlier_score(self, review_sentiment: float, product_id: str, 
                              category: str = None) -> float:
        """Calculate sentiment outlier score using Z-score"""
        try:
            # Get product sentiment history
            if product_id not in self.product_sentiments:
                self.product_sentiments[product_id] = []
            
            # Add current sentiment to history
            self.product_sentiments[product_id].append(review_sentiment)
            
            # Calculate Z-score
            sentiments = self.product_sentiments[product_id]
            if len(sentiments) < 2:
                return 0.0  # Not enough data
                
            mean_sentiment = np.mean(sentiments[:-1])  # Exclude current review
            std_sentiment = np.std(sentiments[:-1])
            
            if std_sentiment == 0:
                return 0.0
                
            z_score = abs(review_sentiment - mean_sentiment) / std_sentiment
            return float(z_score)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment outlier: {e}")
            return 0.0


class BurstinessAnalyzer:
    """Detects review burstiness patterns"""
    
    def __init__(self, time_window_days: int = 7):
        self.time_window = timedelta(days=time_window_days)
        self.product_reviews = {}
        
    def add_review(self, product_id: str, review_time: datetime):
        """Add review timestamp to product history"""
        if product_id not in self.product_reviews:
            self.product_reviews[product_id] = []
        self.product_reviews[product_id].append(review_time)
        
    def calculate_burst_score(self, product_id: str, review_time: datetime) -> float:
        """Calculate burstiness score for a review"""
        try:
            if product_id not in self.product_reviews:
                return 1.0  # Normal activity
                
            # Get reviews in time window
            window_start = review_time - self.time_window
            window_end = review_time + self.time_window
            
            reviews_in_window = [
                t for t in self.product_reviews[product_id]
                if window_start <= t <= window_end and t != review_time
            ]
            
            # Calculate burst score
            if len(reviews_in_window) == 0:
                return 1.0  # Normal activity
                
            # Higher score = more bursty
            burst_score = len(reviews_in_window) + 1
            return float(burst_score)
            
        except Exception as e:
            logger.error(f"Error calculating burst score: {e}")
            return 1.0


class TemplateDetector:
    """Detects templated or repetitive review patterns"""
    
    def __init__(self, ngram_range: Tuple[int, int] = (2, 4)):
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.corpus_vectors = None
        self.corpus_texts = []
        
    def add_to_corpus(self, text: str):
        """Add text to corpus for template detection"""
        if text and len(text.strip()) > 10:
            self.corpus_texts.append(text)
            
    def fit_corpus(self):
        """Fit TF-IDF on corpus"""
        if len(self.corpus_texts) < 2:
            return
            
        try:
            self.corpus_vectors = self.tfidf.fit_transform(self.corpus_texts)
        except Exception as e:
            logger.error(f"Error fitting TF-IDF corpus: {e}")
            
    def calculate_template_score(self, text: str) -> float:
        """Calculate template/redundancy score"""
        try:
            if not self.corpus_vectors or not text:
                return 0.5  # Neutral score
                
            # Transform input text
            text_vector = self.tfidf.transform([text])
            
            # Calculate similarity with corpus
            similarities = cosine_similarity(text_vector, self.corpus_vectors)[0]
            
            # Higher max similarity = more templated
            max_similarity = np.max(similarities)
            
            # Convert to uniqueness score (1 = unique, 0 = very templated)
            uniqueness_score = 1.0 - max_similarity
            return float(uniqueness_score)
            
        except Exception as e:
            logger.error(f"Error calculating template score: {e}")
            return 0.5


class TextAnalysisPipeline:
    """Main text analysis pipeline combining all modules"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.semantic_analyzer = SemanticCoherenceAnalyzer(
            config['features']['semantic_coherence']['model_name']
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.burstiness_analyzer = BurstinessAnalyzer(
            config['features']['burstiness']['time_window_days']
        )
        self.template_detector = TemplateDetector(
            tuple(config['features']['template_detection']['ngram_range'])
        )
        
    def analyze_review(self, review_data: Dict, product_data: Dict) -> Dict:
        """Analyze a single review and return all text-based features"""
        
        review_text = review_data.get('reviewText', '')
        product_description = product_data.get('description', '')
        product_title = product_data.get('title', '')
        
        # Combine product text
        product_text = f"{product_title} {product_description}".strip()
        
        # 1. Semantic Coherence
        coherence_score = self.semantic_analyzer.calculate_coherence(
            review_text, product_text
        )
        
        # 2. Sentiment Analysis
        sentiment_polarity = self.sentiment_analyzer.get_sentiment_polarity(review_text)
        outlier_score = self.sentiment_analyzer.calculate_outlier_score(
            sentiment_polarity, 
            review_data.get('asin'),
            product_data.get('categories', [None])[0] if product_data.get('categories') else None
        )
        
        # 3. Burstiness
        review_time = datetime.fromtimestamp(review_data.get('unixReviewTime', 0))
        self.burstiness_analyzer.add_review(review_data.get('asin'), review_time)
        burst_score = self.burstiness_analyzer.calculate_burst_score(
            review_data.get('asin'), review_time
        )
        
        # 4. Template Detection
        self.template_detector.add_to_corpus(review_text)
        template_score = self.template_detector.calculate_template_score(review_text)
        
        return {
            'semantic_coherence': coherence_score,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_outlier_score': outlier_score,
            'burst_score': burst_score,
            'template_score': template_score
        }
        
    def finalize_analysis(self):
        """Finalize analysis (fit template detector)"""
        self.template_detector.fit_corpus() 