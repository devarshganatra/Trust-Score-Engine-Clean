#!/usr/bin/env python3
"""
Deployment Script for Trust Score Engine
Sets up environment, installs dependencies, and runs initial tests
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a step with description"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
            return True
        else:
            print("❌ Failed")
            print("Error:")
            print(result.stderr)
            if check:
                sys.exit(1)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if check:
            sys.exit(1)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print("✅ Python version is compatible")

def create_virtual_environment():
    """Create virtual environment"""
    print_step(2, "Creating Virtual Environment")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        return
    
    run_command("python -m venv venv", "Creating virtual environment")
    
    # Activate virtual environment
    if platform.system() == "Windows":
        activate_script = "venv\\Scripts\\activate"
    else:
        activate_script = "source venv/bin/activate"
    
    print(f"Virtual environment created. Activate with: {activate_script}")

def install_dependencies():
    """Install project dependencies"""
    print_step(3, "Installing Dependencies")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    run_command("pip install -r requirements.txt", "Installing project dependencies")
    
    # Install development dependencies
    run_command("pip install pytest pytest-cov black flake8", "Installing development dependencies")

def create_directories():
    """Create necessary directories"""
    print_step(4, "Creating Project Directories")
    
    directories = [
        "data",
        "output", 
        "logs",
        "models",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_initial_tests():
    """Run initial tests to verify installation"""
    print_step(5, "Running Initial Tests")
    
    # Test imports
    test_imports = """
import sys
sys.path.append('src')

try:
    from pipeline.trust_score_pipeline import TrustScorePipeline
    from modules.text_analysis import TextAnalysisPipeline
    from modules.reviewer_profiling import ReviewerProfiler
    from models.trust_score_model import TrustScoreModel
    from utils.evaluation import TrustScoreEvaluator
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    run_command(f'python -c "{test_imports}"', "Testing imports")

def generate_sample_data():
    """Generate sample data for testing"""
    print_step(6, "Generating Sample Data")
    
    run_command("python demo.py", "Running demo to generate sample data")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print_step(7, "Running Comprehensive Tests")
    
    run_command("python run_tests.py", "Running test suite", check=False)

def show_next_steps():
    """Show next steps for the user"""
    print_header("Deployment Complete!")
    
    print("🎉 Trust Score Engine has been successfully deployed!")
    
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run the demo:")
    print("   python demo.py")
    
    print("\n3. Start the dashboard:")
    print("   streamlit run dashboard.py")
    
    print("\n4. Process your own data:")
    print("   python main.py run --reviews data/your_reviews.json --products data/your_products.json")
    
    print("\n5. Run tests:")
    print("   python run_tests.py")
    
    print("\n📁 Project Structure:")
    print("   ├── src/           # Source code")
    print("   ├── tests/          # Test suite")
    print("   ├── data/           # Data files")
    print("   ├── output/         # Results and reports")
    print("   ├── config/         # Configuration")
    print("   ├── demo.py         # Interactive demo")
    print("   ├── dashboard.py    # Web dashboard")
    print("   └── main.py         # CLI interface")
    
    print("\n🔗 Useful Links:")
    print("   - Technical Deep Dive: TECHNICAL_DEEP_DIVE.md")
    print("   - Installation Guide: INSTALL.md")
    print("   - Test Results: output/test_report.txt")
    print("   - Performance Report: output/demo_performance_report.md")

def main():
    """Main deployment function"""
    print_header("Trust Score Engine Deployment")
    
    print("This script will:")
    print("1. Check Python version compatibility")
    print("2. Create virtual environment")
    print("3. Install all dependencies")
    print("4. Create project directories")
    print("5. Run initial tests")
    print("6. Generate sample data")
    print("7. Run comprehensive tests")
    
    response = input("\nContinue with deployment? (y/N): ")
    if response.lower() != 'y':
        print("Deployment cancelled.")
        sys.exit(0)
    
    try:
        check_python_version()
        create_virtual_environment()
        install_dependencies()
        create_directories()
        run_initial_tests()
        generate_sample_data()
        run_comprehensive_tests()
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 