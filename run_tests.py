#!/usr/bin/env python3
"""
Test Runner for Trust Score Engine
Runs comprehensive tests and generates reports
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({end_time - start_time:.2f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ FAILED ({end_time - start_time:.2f}s)")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all tests and generate reports"""
    
    print("🧪 Trust Score Engine - Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Test results
    results = []
    
    # 1. Run unit tests
    success = run_command(
        "python -m pytest tests/ -v --tb=short",
        "Running Unit Tests"
    )
    results.append(("Unit Tests", success))
    
    # 2. Run tests with coverage
    success = run_command(
        "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing",
        "Running Tests with Coverage"
    )
    results.append(("Test Coverage", success))
    
    # 3. Run demo
    success = run_command(
        "python demo.py",
        "Running Demo"
    )
    results.append(("Demo", success))
    
    # 4. Check code style
    success = run_command(
        "python -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503",
        "Checking Code Style"
    )
    results.append(("Code Style", success))
    
    # 5. Run integration test
    success = run_command(
        "python -c \"import sys; sys.path.append('src'); from tests.test_trust_engine import TestIntegration; t = TestIntegration(); t.test_end_to_end_pipeline()\"",
        "Running Integration Test"
    )
    results.append(("Integration Test", success))
    
    # Generate test report
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Save report
    with open('output/test_report.txt', 'w') as f:
        f.write("Trust Score Engine - Test Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for test_name, success in results:
            status = "PASS" if success else "FAIL"
            f.write(f"{test_name}: {status}\n")
        
        f.write(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)\n")
    
    print(f"\n📄 Test report saved to: output/test_report.txt")
    
    # Exit with appropriate code
    if passed == total:
        print("\n🎉 All tests passed! The system is ready for deployment.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review and fix issues.")
        sys.exit(1)

if __name__ == "__main__":
    main() 