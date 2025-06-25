#!/usr/bin/env python3
"""
Trust Score Engine - Main Entry Point
Command-line interface for the Trust Score Engine pipeline
"""

import argparse
import logging
import yaml
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from pipeline.trust_score_pipeline import TrustScorePipeline

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trust_engine.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path='config/pipeline_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def run_pipeline(config, reviews_file, products_file, output_file=None):
    """Run the complete trust score pipeline"""
    print("🚀 Starting Trust Score Engine Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = TrustScorePipeline(config)
    
    try:
        # Load data
        print("📂 Loading data...")
        if not pipeline.load_data(reviews_file, products_file):
            print("❌ Failed to load data")
            return False
        
        # Process reviews
        print("🔍 Processing reviews...")
        if not pipeline.process_reviews():
            print("❌ Failed to process reviews")
            return False
        
        # Train model
        print("🤖 Training trust score model...")
        if not pipeline.train_model():
            print("⚠️  Model training failed, using heuristic scoring")
        
        # Save results
        if output_file:
            print("💾 Saving results...")
            pipeline.save_results(output_file)
        
        # Print statistics
        print("\n📊 Pipeline Statistics:")
        print("-" * 30)
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False
    finally:
        pipeline.cleanup()

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🌐 Starting Trust Score Engine Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    
    import subprocess
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Trust Score Engine - Review Credibility Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --reviews data/reviews.json --products data/products.json
  python main.py dashboard
  python main.py run --config custom_config.yaml --output results.csv
        """
    )
    
    # Add verbose argument to main parser
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run the trust score pipeline')
    run_parser.add_argument('--reviews', required=True, help='Path to reviews JSON file')
    run_parser.add_argument('--products', required=True, help='Path to products JSON file')
    run_parser.add_argument('--config', default='config/pipeline_config.yaml', help='Configuration file path')
    run_parser.add_argument('--output', help='Output CSV file path')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the Streamlit dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.command == 'run':
        # Load configuration
        config = load_config(args.config)
        
        # Set output file
        output_file = args.output or f"output/processed_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Run pipeline
        success = run_pipeline(config, args.reviews, args.products, output_file)
        sys.exit(0 if success else 1)
    
    elif args.command == 'dashboard':
        run_dashboard()

if __name__ == "__main__":
    main() 