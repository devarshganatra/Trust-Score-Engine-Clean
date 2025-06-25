"""
Rewards System Module
Manages points allocation and voucher redemption based on trust scores
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RewardsSystem:
    """Rewards system for trust score engine"""
    
    def __init__(self, config: Dict, database):
        self.config = config
        self.db = database
        self.rewards_config = config['rewards']
        
    def calculate_points(self, trust_score: float) -> int:
        """Calculate points to award based on trust score"""
        try:
            points_per_score = self.rewards_config['points_per_trust_score']
            points = int(trust_score * points_per_score)
            return max(points, 0)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating points: {e}")
            return 0
    
    def award_points(self, reviewer_id: str, trust_score: float, review_id: str) -> Dict:
        """Award points to reviewer for a high-trust review"""
        try:
            points = self.calculate_points(trust_score)
            
            if points <= 0:
                return {'success': False, 'message': 'No points awarded for low trust score'}
            
            # Create reward record
            reward_data = {
                'reward_id': str(uuid.uuid4()),
                'reviewerID': reviewer_id,
                'review_id': review_id,
                'trust_score': trust_score,
                'points_awarded': points,
                'reward_type': 'trust_score_points',
                'status': 'awarded',
                'created_at': datetime.now()
            }
            
            # Save to database
            success = self.db.save_reward(reward_data)
            
            if success:
                logger.info(f"Awarded {points} points to reviewer {reviewer_id} for trust score {trust_score}")
                return {
                    'success': True,
                    'points_awarded': points,
                    'reward_id': reward_data['reward_id'],
                    'message': f'Awarded {points} points for high trust score'
                }
            else:
                return {'success': False, 'message': 'Failed to save reward to database'}
                
        except Exception as e:
            logger.error(f"Error awarding points: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def get_reviewer_points(self, reviewer_id: str) -> int:
        """Get total points for a reviewer"""
        try:
            rewards = self.db.get_reviewer_rewards(reviewer_id)
            
            total_points = sum(
                reward['points_awarded'] 
                for reward in rewards 
                if reward.get('status') == 'awarded'
            )
            
            return total_points
            
        except Exception as e:
            logger.error(f"Error getting reviewer points: {e}")
            return 0
    
    def get_available_vouchers(self, points: int) -> List[Dict]:
        """Get available vouchers based on points"""
        try:
            voucher_values = self.rewards_config['voucher_values']
            min_points = self.rewards_config['min_points_for_redemption']
            
            if points < min_points:
                return []
            
            available_vouchers = []
            
            for voucher_type, value in voucher_values.items():
                if points >= value:
                    available_vouchers.append({
                        'type': voucher_type,
                        'value': value,
                        'description': f'${value} voucher'
                    })
            
            return available_vouchers
            
        except Exception as e:
            logger.error(f"Error getting available vouchers: {e}")
            return []
    
    def redeem_voucher(self, reviewer_id: str, voucher_type: str) -> Dict:
        """Redeem voucher for points"""
        try:
            # Get reviewer's total points
            total_points = self.get_reviewer_points(reviewer_id)
            
            # Get voucher value
            voucher_values = self.rewards_config['voucher_values']
            if voucher_type not in voucher_values:
                return {'success': False, 'message': 'Invalid voucher type'}
            
            voucher_value = voucher_values[voucher_type]
            
            # Check if enough points
            if total_points < voucher_value:
                return {
                    'success': False, 
                    'message': f'Insufficient points. Need {voucher_value}, have {total_points}'
                }
            
            # Create redemption record
            redemption_data = {
                'reward_id': str(uuid.uuid4()),
                'reviewerID': reviewer_id,
                'voucher_type': voucher_type,
                'voucher_value': voucher_value,
                'points_used': voucher_value,
                'reward_type': 'voucher_redemption',
                'status': 'redeemed',
                'created_at': datetime.now()
            }
            
            # Save to database
            success = self.db.save_reward(redemption_data)
            
            if success:
                logger.info(f"Redeemed {voucher_type} voucher for reviewer {reviewer_id}")
                return {
                    'success': True,
                    'voucher_type': voucher_type,
                    'voucher_value': voucher_value,
                    'points_used': voucher_value,
                    'remaining_points': total_points - voucher_value,
                    'message': f'Successfully redeemed ${voucher_value} voucher'
                }
            else:
                return {'success': False, 'message': 'Failed to save redemption to database'}
                
        except Exception as e:
            logger.error(f"Error redeeming voucher: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def get_reviewer_summary(self, reviewer_id: str) -> Dict:
        """Get comprehensive summary for a reviewer"""
        try:
            # Get total points
            total_points = self.get_reviewer_points(reviewer_id)
            
            # Get available vouchers
            available_vouchers = self.get_available_vouchers(total_points)
            
            # Get reward history
            rewards = self.db.get_reviewer_rewards(reviewer_id)
            
            # Calculate statistics
            total_rewards = len(rewards)
            total_points_earned = sum(
                reward['points_awarded'] 
                for reward in rewards 
                if reward.get('reward_type') == 'trust_score_points'
            )
            total_vouchers_redeemed = sum(
                1 for reward in rewards 
                if reward.get('reward_type') == 'voucher_redemption'
            )
            
            return {
                'reviewer_id': reviewer_id,
                'total_points': total_points,
                'total_points_earned': total_points_earned,
                'total_rewards': total_rewards,
                'total_vouchers_redeemed': total_vouchers_redeemed,
                'available_vouchers': available_vouchers,
                'can_redeem': len(available_vouchers) > 0,
                'recent_rewards': rewards[:10]  # Last 10 rewards
            }
            
        except Exception as e:
            logger.error(f"Error getting reviewer summary: {e}")
            return {
                'reviewer_id': reviewer_id,
                'total_points': 0,
                'total_points_earned': 0,
                'total_rewards': 0,
                'total_vouchers_redeemed': 0,
                'available_vouchers': [],
                'can_redeem': False,
                'recent_rewards': []
            }
    
    def get_top_reviewers(self, limit: int = 10) -> List[Dict]:
        """Get top reviewers by points earned"""
        try:
            # This would require a more complex database query
            # For now, we'll get all reviewers and sort by points
            # In a real implementation, you'd want to optimize this with aggregation
            
            # Get all reviewers (this is simplified - in production you'd use aggregation)
            all_reviewers = []
            
            # Get unique reviewer IDs from rewards
            rewards = self.db.collections['rewards'].find({})
            reviewer_ids = set()
            
            for reward in rewards:
                reviewer_ids.add(reward['reviewerID'])
            
            # Get summary for each reviewer
            for reviewer_id in list(reviewer_ids)[:limit * 2]:  # Get more to account for filtering
                summary = self.get_reviewer_summary(reviewer_id)
                if summary['total_points'] > 0:  # Only include reviewers with points
                    all_reviewers.append(summary)
            
            # Sort by total points and return top
            top_reviewers = sorted(
                all_reviewers, 
                key=lambda x: x['total_points'], 
                reverse=True
            )[:limit]
            
            return top_reviewers
            
        except Exception as e:
            logger.error(f"Error getting top reviewers: {e}")
            return []
    
    def process_trust_score_reward(self, review_data: Dict, trust_score: float) -> Dict:
        """Process reward for a review based on trust score"""
        try:
            reviewer_id = review_data.get('reviewerID')
            review_id = f"{reviewer_id}_{review_data.get('asin')}"
            
            # Check if trust score meets threshold
            threshold = self.config['trust_score']['threshold_for_rewards']
            
            if trust_score >= threshold:
                return self.award_points(reviewer_id, trust_score, review_id)
            else:
                return {
                    'success': False, 
                    'message': f'Trust score {trust_score} below threshold {threshold}'
                }
                
        except Exception as e:
            logger.error(f"Error processing trust score reward: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'} 