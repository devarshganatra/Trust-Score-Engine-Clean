# Trust Score Engine - Installation Guide

## 🚀 Quick Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **8GB+ RAM** (for processing large datasets)
- **MongoDB** (optional, for data persistence)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/trust-score-engine.git
cd trust-score-engine
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Quick Start (Optional)

```bash
# Run the quick start script to test the installation
python quick_start.py
```

## 📦 Detailed Installation

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+

#### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 50GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

### Dependencies Breakdown

#### Core ML Libraries
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting for trust score fusion

#### NLP Libraries
- **Transformers**: BERT models for semantic analysis
- **Torch**: PyTorch for deep learning
- **TextBlob**: Sentiment analysis
- **Sentence-Transformers**: Text embeddings

#### Database
- **PyMongo**: MongoDB driver
- **Motor**: Async MongoDB driver

#### Visualization
- **Streamlit**: Web dashboard
- **Plotly**: Interactive charts
- **Altair**: Statistical visualizations

### MongoDB Setup (Optional)

#### Option 1: Local Installation

**Ubuntu/Debian:**
```bash
# Install MongoDB
sudo apt update
sudo apt install mongodb

# Start MongoDB service
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

**macOS:**
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb-community
```

**Windows:**
1. Download MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Install and start the MongoDB service

#### Option 2: Docker

```bash
# Pull MongoDB image
docker pull mongo:latest

# Run MongoDB container
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

#### Option 3: MongoDB Atlas (Cloud)

1. Create account at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Get connection string and update `config/pipeline_config.yaml`

### GPU Support (Optional)

For faster processing with GPU acceleration:

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CUDA-enabled transformers
pip install transformers[torch]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=trust_engine

# Model Configuration
MODEL_CACHE_DIR=./models
TRANSFORMERS_CACHE_DIR=./cache

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trust_engine.log

# Processing
BATCH_SIZE=1000
MAX_WORKERS=4
```

### Configuration File

Edit `config/pipeline_config.yaml` to customize:

```yaml
# Database settings
database:
  mongodb_uri: "mongodb://localhost:27017/"
  database_name: "trust_engine"

# Model parameters
model:
  fusion_model: "xgboost"
  xgboost_params:
    n_estimators: 100
    max_depth: 6

# Trust score settings
trust_score:
  threshold_for_rewards: 70
  normalization_method: "min_max"

# Rewards system
rewards:
  points_per_trust_score: 1.0
  min_points_for_redemption: 50
```

## 🧪 Testing the Installation

### 1. Run Quick Start

```bash
python quick_start.py
```

This will:
- Check dependencies
- Create sample data
- Run the pipeline
- Show results

### 2. Test Individual Components

```bash
# Test text analysis
python -c "
from src.modules.text_analysis import TextAnalysisPipeline
import yaml
config = yaml.safe_load(open('config/pipeline_config.yaml'))
analyzer = TextAnalysisPipeline(config)
print('✅ Text analysis module working')
"

# Test database connection
python -c "
from src.utils.database import TrustEngineDatabase
import yaml
config = yaml.safe_load(open('config/pipeline_config.yaml'))
db = TrustEngineDatabase(config)
print('✅ Database connection working')
"
```

### 3. Run Dashboard

```bash
python main.py dashboard
```

Visit `http://localhost:8501` to see the dashboard.

## 📊 Data Preparation

### Review Data Format

Your review data should be in JSON format with one review per line:

```json
{
  "reviewerID": "A1B2C3D4E5F6",
  "asin": "B000123456",
  "reviewText": "This product is amazing!",
  "overall": 5,
  "verified": true,
  "helpful": [10, 12],
  "unixReviewTime": 1609459200
}
```

### Product Data Format

Your product data should be in JSON format with one product per line:

```json
{
  "asin": "B000123456",
  "title": "Product Title",
  "description": "Product description...",
  "categories": [["Electronics", "Audio"]],
  "brand": "Brand Name",
  "price": 99.99
}
```

## 🚀 Running the Pipeline

### Basic Usage

```bash
# Run with your data
python main.py run \
  --reviews data/your_reviews.json \
  --products data/your_products.json \
  --output results.csv
```

### Advanced Usage

```bash
# Run with custom configuration
python main.py run \
  --reviews data/reviews.json \
  --products data/products.json \
  --config custom_config.yaml \
  --output results.csv \
  --verbose
```

### Batch Processing

For large datasets, you can process in batches:

```bash
# Process first 10,000 reviews
python main.py run \
  --reviews data/reviews.json \
  --products data/products.json \
  --output batch1.csv
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: Out of memory errors
**Solution**: 
- Reduce batch size in config
- Use smaller datasets for testing
- Increase system RAM

#### 2. MongoDB Connection Issues
**Problem**: Cannot connect to MongoDB
**Solution**:
- Check if MongoDB is running
- Verify connection string
- Use local file storage instead

#### 3. Model Download Issues
**Problem**: BERT models not downloading
**Solution**:
- Check internet connection
- Clear transformers cache: `rm -rf ~/.cache/huggingface`
- Use VPN if needed

#### 4. CUDA Issues
**Problem**: GPU not being used
**Solution**:
- Install CUDA-enabled PyTorch
- Check GPU drivers
- Set `CUDA_VISIBLE_DEVICES=0`

### Performance Optimization

#### For Large Datasets
1. **Increase batch size** in config
2. **Use GPU acceleration**
3. **Process in parallel** with multiple workers
4. **Use SSD storage** for faster I/O

#### For Real-time Processing
1. **Pre-load models** in memory
2. **Use model caching**
3. **Optimize database queries**
4. **Use async processing**

## 📚 Next Steps

After installation:

1. **Add your data** to the `data/` directory
2. **Customize configuration** in `config/pipeline_config.yaml`
3. **Run the pipeline** with your data
4. **Monitor results** using the dashboard
5. **Fine-tune models** based on your specific use case

## 🆘 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the logs in `logs/trust_engine.log`
3. Open an issue on GitHub
4. Check the documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 