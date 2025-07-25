"""
Trust Score Engine Dashboard
Streamlit-based monitoring and visualization interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import yaml
from textblob import TextBlob
import sys
import os

# Add src to path for imports
sys.path.append('src')

from utils.database import TrustEngineDatabase
from modules.rewards_system import RewardsSystem

# Page configuration
st.set_page_config(
    page_title="Trust Score Engine Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize database connection
@st.cache_resource
def get_database():
    return TrustEngineDatabase(config)

@st.cache_resource
def get_rewards_system():
    db = get_database()
    return RewardsSystem(config, db)

db = get_database()
rewards_system = get_rewards_system()

# Sidebar
st.sidebar.title("🔍 Trust Score Engine")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["📊 Overview", "📈 Trust Scores", "👥 Reviewers", "🚨 Alerts", "⚙️ Settings", "📝 Test a Review"]
)

# Main content
if page == "📊 Overview":
    st.title("📊 Trust Score Engine Overview")
    
    # Get statistics
    stats = db.get_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reviews",
            value=f"{stats.get('total_reviews', 0):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Reviewers",
            value=f"{stats.get('total_reviewers', 0):,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Avg Trust Score",
            value=f"{stats.get('avg_trust_score', 0):.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="High Trust Reviews",
            value=f"{stats.get('high_trust_reviews', 0):,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Trust Score Distribution
    st.subheader("📈 Trust Score Distribution")
    
    # Get trust scores for visualization
    trust_scores = db.get_all_trust_scores()
    if trust_scores:
        scores = [ts['trust_score'] for ts in trust_scores if 'trust_score' in ts]
        
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Distribution of Trust Scores",
            labels={'x': 'Trust Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Score", f"{np.mean(scores):.2f}")
        with col2:
            st.metric("Median Score", f"{np.median(scores):.2f}")
        with col3:
            st.metric("Std Deviation", f"{np.std(scores):.2f}")
    
    # Recent Activity
    st.subheader("🕒 Recent Activity")
    
    # Get recent trust scores
    recent_scores = sorted(db.get_all_trust_scores(), key=lambda x: x.get('created_at', ''), reverse=True)[:10]
    
    if recent_scores:
        recent_df = pd.DataFrame(recent_scores)
        if 'created_at' in recent_df:
            recent_df['created_at'] = pd.to_datetime(recent_df['created_at'])
            
            fig = px.line(
                recent_df,
                x='created_at',
                y='trust_score',
                title="Recent Trust Scores",
                labels={'created_at': 'Time', 'trust_score': 'Trust Score'}
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Trust Scores":
    st.title("📈 Trust Score Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Trust Score", 0, 100, 0)
    
    with col2:
        max_score = st.slider("Maximum Trust Score", 0, 100, 100)
    
    with col3:
        limit = st.selectbox("Number of Results", [50, 100, 200, 500])
    
    # Get filtered trust scores
    all_trust_scores = db.get_all_trust_scores()
    trust_scores = [ts for ts in all_trust_scores if min_score <= ts.get('trust_score', 0) <= max_score]
    trust_scores = sorted(trust_scores, key=lambda x: x.get('trust_score', 0), reverse=True)[:limit]
    
    if trust_scores:
        df = pd.DataFrame(trust_scores)
        
        # Display table
        st.subheader("Trust Score Details")
        st.dataframe(
            df[['reviewerID', 'asin', 'trust_score', 'created_at']].head(20),
            use_container_width=True
        )
        
        # Score analysis
        st.subheader("Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score ranges
            score_ranges = {
                'Very Low (0-20)': len([ts for ts in trust_scores if ts['trust_score'] <= 20]),
                'Low (21-40)': len([ts for ts in trust_scores if 21 <= ts['trust_score'] <= 40]),
                'Medium (41-60)': len([ts for ts in trust_scores if 41 <= ts['trust_score'] <= 60]),
                'High (61-80)': len([ts for ts in trust_scores if 61 <= ts['trust_score'] <= 80]),
                'Very High (81-100)': len([ts for ts in trust_scores if ts['trust_score'] >= 81])
            }
            
            fig = px.pie(
                values=list(score_ranges.values()),
                names=list(score_ranges.keys()),
                title="Trust Score Distribution by Range"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance (if available)
            st.subheader("Feature Analysis")
            
            # Sample feature data
            if 'text_features' in df.columns and len(df) > 0:
                sample_features = df.iloc[0]['text_features']
                if sample_features:
                    feature_names = list(sample_features.keys())
                    feature_values = list(sample_features.values())
                    
                    fig = px.bar(
                        x=feature_names,
                        y=feature_values,
                        title="Sample Text Features",
                        labels={'x': 'Feature', 'y': 'Value'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == "👥 Reviewers":
    st.title("👥 Reviewer Analysis")
    
    # Reviewer search
    reviewer_id = st.text_input("Enter Reviewer ID to analyze:")
    
    if reviewer_id:
        # Get reviewer profile
        profile = db.get_reviewer_profile(reviewer_id)
        reviews = db.get_reviewer_reviews(reviewer_id)
        
        if profile:
            st.subheader(f"Reviewer Profile: {reviewer_id}")
            
            # Profile metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Verified Purchase Ratio", f"{profile.get('verified_purchase_ratio', 0):.2f}")
            
            with col2:
                st.metric("Helpfulness Ratio", f"{profile.get('helpfulness_ratio', 0):.2f}")
            
            with col3:
                st.metric("Activity Pattern Score", f"{profile.get('activity_pattern_score', 0):.2f}")
            
            with col4:
                st.metric("Composite Score", f"{profile.get('composite_reviewer_score', 0):.2f}")
            
            # Reviewer's reviews
            if reviews:
                st.subheader("Reviewer's Reviews")
                
                reviews_df = pd.DataFrame(reviews)
                if 'trust_score' in reviews_df.columns:
                    fig = px.scatter(
                        reviews_df,
                        x='unixReviewTime',
                        y='trust_score',
                        title="Trust Scores Over Time",
                        labels={'unixReviewTime': 'Review Time', 'trust_score': 'Trust Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Reviews table
                st.dataframe(
                    reviews_df[['asin', 'overall', 'verified', 'unixReviewTime']].head(10),
                    use_container_width=True
                )
        else:
            st.warning("Reviewer not found")
    
    # Top reviewers
    st.subheader("🏆 Top Reviewers by Trust Score")
    
    # Get top reviewers (simplified - in production you'd optimize this)
    top_reviewers = rewards_system.get_top_reviewers(10)
    
    if top_reviewers:
        top_df = pd.DataFrame(top_reviewers)
        
        fig = px.bar(
            top_df,
            x='reviewer_id',
            y='total_points',
            title="Top Reviewers by Points Earned",
            labels={'reviewer_id': 'Reviewer ID', 'total_points': 'Total Points'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            top_df[['reviewer_id', 'total_points', 'total_rewards', 'total_vouchers_redeemed']],
            use_container_width=True
        )

# Rewards page removed
    st.title("🎁 Rewards System")
    
    # Rewards overview
    rewards_stats = db.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rewards", f"{rewards_stats.get('total_rewards', 0):,}")
    
    with col2:
        st.metric("Points Awarded", f"{rewards_stats.get('total_rewards', 0) * 70:,}")  # Estimate
    
    with col3:
        st.metric("Vouchers Redeemed", f"{rewards_stats.get('total_rewards', 0) // 10:,}")  # Estimate
    
    st.markdown("---")
    
    # Voucher redemption
    st.subheader("💳 Voucher Redemption")
    
    reviewer_id = st.text_input("Enter Reviewer ID for voucher redemption:")
    
    if reviewer_id:
        summary = rewards_system.get_reviewer_summary(reviewer_id)
        
        if summary['total_points'] > 0:
            st.success(f"Reviewer has {summary['total_points']} points available")
            
            # Available vouchers
            if summary['available_vouchers']:
                st.subheader("Available Vouchers:")
                
                for voucher in summary['available_vouchers']:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{voucher['description']}**")
                    
                    with col2:
                        st.write(f"Cost: {voucher['value']} points")
                    
                    with col3:
                        if st.button(f"Redeem {voucher['type']}", key=voucher['type']):
                            result = rewards_system.redeem_voucher(reviewer_id, voucher['type'])
                            if result['success']:
                                st.success(result['message'])
                                st.rerun()
                            else:
                                st.error(result['message'])
            else:
                st.info("No vouchers available for redemption")
                
            # Reward history
            if summary['recent_rewards']:
                st.subheader("Recent Rewards")
                
                rewards_df = pd.DataFrame(summary['recent_rewards'])
                st.dataframe(
                    rewards_df[['reward_type', 'points_awarded', 'created_at']].head(10),
                    use_container_width=True
                )
        else:
            st.warning("Reviewer has no points available")
    
    # Rewards distribution
    st.subheader("📊 Rewards Distribution")
    
    # Get all rewards
    all_rewards = list(db.collections['rewards'].find())
    
    if all_rewards:
        rewards_df = pd.DataFrame(all_rewards)
        
        # Rewards by type
        reward_types = rewards_df['reward_type'].value_counts()
        
        fig = px.pie(
            values=reward_types.values,
            names=reward_types.index,
            title="Rewards by Type"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🚨 Alerts":
    st.title("🚨 Suspicious Review Alerts")
    
    # Suspicious reviews threshold
    threshold = st.slider("Suspicious Score Threshold", 0, 50, 30)
    
    # Get suspicious reviews
    suspicious_reviews = db.get_suspicious_reviews(threshold, 50)
    
    if suspicious_reviews:
        st.subheader(f"Reviews with Trust Score < {threshold}")
        
        suspicious_df = pd.DataFrame(suspicious_reviews)
        
        # Display suspicious reviews
        for review in suspicious_reviews[:10]:
            with st.expander(f"Review by {review.get('reviewerID', 'Unknown')} - Score: {review.get('trust_score', 0):.1f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Review Details:**")
                    st.write(f"Product: {review.get('asin', 'Unknown')}")
                    st.write(f"Trust Score: {review.get('trust_score', 0):.1f}")
                    st.write(f"Date: {review.get('created_at', 'Unknown')}")
                
                with col2:
                    st.write("**Features:**")
                    if 'text_features' in review:
                        features = review['text_features']
                        st.write(f"Semantic Coherence: {features.get('semantic_coherence', 0):.2f}")
                        st.write(f"Template Score: {features.get('template_score', 0):.2f}")
                        st.write(f"Burst Score: {features.get('burst_score', 0):.2f}")
        
        # Suspicious patterns
        st.subheader("🔍 Suspicious Patterns Analysis")
        
        if len(suspicious_reviews) > 0:
            # Low template scores
            low_template = [r for r in suspicious_reviews if r.get('text_features', {}).get('template_score', 1) < 0.3]
            st.metric("Low Template Score Reviews", len(low_template))
            
            # High burst scores
            high_burst = [r for r in suspicious_reviews if r.get('text_features', {}).get('burst_score', 1) > 3]
            st.metric("High Burst Score Reviews", len(high_burst))
            
            # Low semantic coherence
            low_coherence = [r for r in suspicious_reviews if r.get('text_features', {}).get('semantic_coherence', 1) < 0.3]
            st.metric("Low Semantic Coherence Reviews", len(low_coherence))
    else:
        st.success("No suspicious reviews found!")

elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")
    
    st.subheader("Configuration")
    
    # Display current configuration
    st.json(config)
    
    st.subheader("Database Status")
    
    # Test database connection
    try:
        stats = db.get_statistics()
        st.success("✅ Database connection successful")
        st.write("Database Statistics:", stats)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
    
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Configuration:**")
        st.write(f"Fusion Model: {config['model']['fusion_model']}")
        st.write(f"Trust Score Range: {config['trust_score']['min_score']}-{config['trust_score']['max_score']}")
        st.write(f"Rewards Threshold: {config['trust_score']['threshold_for_rewards']}")
    
    with col2:
        st.write("**Rewards Configuration:**")
        st.write(f"Points per Trust Score: {config['rewards']['points_per_trust_score']}")
        st.write(f"Min Points for Redemption: {config['rewards']['min_points_for_redemption']}")
        st.write(f"Voucher Values: {config['rewards']['voucher_values']}")

elif page == "📝 Test a Review":
    st.title("📝 Test a Review for Trust Score")
    with st.form("test_review_form"):
        review_text = st.text_area("Review Text", "")
        rating = st.slider("Rating (1-5)", 1, 5, 3)
        helpful_votes = st.number_input("Helpful Votes", min_value=0, value=0)
        total_votes = st.number_input("Total Votes", min_value=0, value=0)
        sentiment_score = st.number_input("Sentiment Score (optional)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        submitted = st.form_submit_button("Get Trust Score")
    if submitted:
        # Feature extraction (mimic pipeline)
        review_length = len(review_text)
        helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0.0
        # --- Simple heuristics for demo ---
        # semantic_coherence: assume longer reviews are more coherent
        semantic_coherence = min(len(review_text) / 300, 1.0)
        # sentiment_outlier_score: absolute polarity difference between rating and sentiment polarity
        polarity = TextBlob(review_text).sentiment.polarity if review_text else 0.0  # -1..1
        sentiment_outlier_score = abs((rating - 3)/2 - polarity)  # heuristic
        # burst_score: shorter reviews get higher burst (1..5)
        burst_score = max(1, 5 - int(len(review_text)/40))
        # template_score: assume templates if review very short
        template_score = 0.2 if len(review_text) < 30 else 0.5
        verified_purchase_ratio = 0.0  # unknown in demo
        activity_pattern_score = 0.5   # placeholder
        sentiment_uniformity_score = 1 - abs(polarity)  # high polarity => low uniformity

        features = [
            semantic_coherence,
            sentiment_outlier_score,
            burst_score,
            template_score,
            verified_purchase_ratio,
            helpfulness_ratio,
            activity_pattern_score,
            sentiment_uniformity_score
        ]
        import numpy as np
        features = np.array(features).reshape(1, -1)
        # Load model and scaler from file
        from models.trust_score_model import TrustScoreModel
        model = TrustScoreModel(config)
        model_path = os.path.join("models", "trained_trust_model.pkl")
        scaler_path = os.path.join("models", "trained_trust_model_scaler.pkl")
        if not os.path.exists(model_path):
            st.error("No trained model found. Please run the pipeline first.")
        else:
            try:
                model.load_model(model_path)
                # Try to load scaler if available
                if os.path.exists(scaler_path):
                    import joblib
                    model.scaler = joblib.load(scaler_path)
                scaled = model.scaler.transform(features)
                trust_score = model.model.predict_proba(scaled)[0][1] * 100
                st.metric("Predicted Trust Score", f"{trust_score:.1f}")
            except Exception as e:
                st.error(f"Error scoring review: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Trust Score Engine Dashboard | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
) 