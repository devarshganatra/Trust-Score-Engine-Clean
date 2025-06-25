#!/usr/bin/env python3
"""
Quick Start Script for Trust Score Engine
Sets up the environment and runs a demo
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def create_sample_data():
    """Create sample data for demonstration"""
    print("📝 Creating sample data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample reviews data
    sample_reviews = [
        {
            "reviewerID": "A1B2C3D4E5F6",
            "asin": "B000123456",
            "reviewText": "This product exceeded my expectations! The quality is outstanding and it works perfectly. Highly recommend for anyone looking for a reliable solution.",
            "overall": 5,
            "verified": True,
            "helpful": [15, 16],
            "unixReviewTime": 1609459200
        },
        {
            "reviewerID": "A1B2C3D4E5F6",
            "asin": "B000123457",
            "reviewText": "Great product, fast delivery, excellent customer service. Will definitely buy again!",
            "overall": 5,
            "verified": True,
            "helpful": [8, 9],
            "unixReviewTime": 1609545600
        },
        {
            "reviewerID": "X9Y8Z7W6V5U4",
            "asin": "B000123456",
            "reviewText": "This is the worst product I've ever used. Complete waste of money. Don't buy it!",
            "overall": 1,
            "verified": False,
            "helpful": [2, 10],
            "unixReviewTime": 1609632000
        },
        {
            "reviewerID": "M1N2O3P4Q5R6",
            "asin": "B000123458",
            "reviewText": "Product is okay, nothing special. Does what it says but could be better.",
            "overall": 3,
            "verified": True,
            "helpful": [3, 5],
            "unixReviewTime": 1609718400
        },
        {
            "reviewerID": "S7T8U9V0W1X2",
            "asin": "B000123456",
            "reviewText": "Amazing product! The best purchase I've made this year. Quality is top-notch and the price is reasonable.",
            "overall": 5,
            "verified": True,
            "helpful": [12, 13],
            "unixReviewTime": 1609804800
        }
    ]
    
    # Sample products data
    sample_products = [
        {
            "asin": "B000123456",
            "title": "Premium Wireless Headphones",
            "description": "High-quality wireless headphones with noise cancellation, 30-hour battery life, and premium sound quality. Perfect for music lovers and professionals.",
            "categories": [["Electronics", "Audio", "Headphones"]],
            "brand": "AudioTech",
            "price": 199.99
        },
        {
            "asin": "B000123457",
            "title": "Smart Fitness Tracker",
            "description": "Advanced fitness tracker with heart rate monitoring, GPS tracking, and sleep analysis. Water-resistant and compatible with all smartphones.",
            "categories": [["Electronics", "Wearables", "Fitness Trackers"]],
            "brand": "FitTech",
            "price": 89.99
        },
        {
            "asin": "B000123458",
            "title": "Portable Bluetooth Speaker",
            "description": "Compact portable speaker with 360-degree sound, 20-hour battery life, and waterproof design. Perfect for outdoor activities.",
            "categories": [["Electronics", "Audio", "Speakers"]],
            "brand": "SoundWave",
            "price": 59.99
        }
    ]
    
    # Save sample data
    with open('data/reviews_Electronics_5.json', 'w') as f:
        for review in sample_reviews:
            f.write(json.dumps(review) + '\n')
    
    with open('data/meta_Electronics.json', 'w') as f:
        for product in sample_products:
            f.write(json.dumps(product) + '\n')
    
    print(f"✅ Created {len(sample_reviews)} sample reviews and {len(sample_products)} sample products")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'transformers',
        'torch', 'textblob', 'sentence-transformers', 'pymongo', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed")
    
    return True

def setup_mongodb():
    """Setup MongoDB connection (optional)"""
    print("🗄️  Setting up MongoDB...")
    
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"⚠️  MongoDB not available: {e}")
        print("💡 You can install MongoDB or the system will work without it")
        return False

def run_demo():
    """Run the trust score engine demo"""
    print("🚀 Running Trust Score Engine Demo...")
    print("=" * 50)
    
    try:
        # Run the pipeline
        result = subprocess.run([
            sys.executable, 'main.py', 'run',
            '--reviews', 'data/reviews_Electronics_5.json',
            '--products', 'data/meta_Electronics.json',
            '--output', 'output/demo_results.csv',
            '--verbose'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            print("\n📊 Results saved to: output/demo_results.csv")
            
            # Show sample results
            if os.path.exists('output/demo_results.csv'):
                import pandas as pd
                df = pd.read_csv('output/demo_results.csv')
                print(f"\n📈 Processed {len(df)} reviews")
                if 'trust_score' in df.columns:
                    print(f"📊 Average trust score: {df['trust_score'].mean():.2f}")
                    print(f"🏆 Highest trust score: {df['trust_score'].max():.2f}")
                    print(f"⚠️  Lowest trust score: {df['trust_score'].min():.2f}")
        else:
            print("❌ Demo failed")
            print("Error output:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Main function for quick start"""
    print("🎯 Trust Score Engine - Quick Start")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup MongoDB (optional)
    setup_mongodb()
    
    # Create sample data
    create_sample_data()
    
    # Run demo
    run_demo()
    
    print("\n🎉 Quick start completed!")
    print("\n📋 Next steps:")
    print("1. Add your own data to the 'data/' directory")
    print("2. Run: python main.py run --reviews data/your_reviews.json --products data/your_products.json")
    print("3. Start dashboard: python main.py dashboard")
    print("4. View results in the 'output/' directory")

if __name__ == "__main__":
    main() 