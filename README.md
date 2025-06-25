# Trust Score Engine Pipeline

A comprehensive review credibility assessment system that analyzes review authenticity and assigns trust scores using multiple AI/ML modules.

## 🏗️ System Architecture

### Core Components:
1. **Text Analysis Modules**
   - Semantic Coherence (DistilBERT)
   - Sentiment Outlier Detection
   - Burstiness Analysis
   - Template/Redundancy Detection

2. **Reviewer Profiling**
   - Verified Purchase Analysis
   - Activity Pattern Recognition
   - Helpful Vote Ratio
   - Sentiment Uniformity

3. **Trust Score Fusion Model**
   - Multi-feature integration
   - XGBoost/LightGBM based scoring
   - 0-100 trust score output

4. **Rewards System**
   - Points allocation based on trust scores
   - Voucher/discount redemption

## 📁 Project Structure

```
trustengine/
├── data/                    # Dataset storage
├── src/
│   ├── modules/            # Core analysis modules
│   ├── models/             # ML models
│   ├── utils/              # Utility functions
│   └── pipeline/           # Main pipeline
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── main.py                # Entry point
```

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Datasets:**
   - Put `reviews_Electronics_5.json` in `data/`
   - Put `meta_Electronics.json` in `data/`

3. **Run Pipeline:**
   ```bash
   python main.py
   ```

## 📊 Features

- **Multi-modal Analysis**: Combines text, temporal, and behavioral features
- **Scalable Architecture**: Modular design for easy extension
- **Real-time Scoring**: Fast inference for new reviews
- **Rewards Integration**: Automated incentive system
- **Monitoring Dashboard**: Streamlit-based visualization

## 🔧 Configuration

Edit `config/pipeline_config.yaml` to customize:
- Model parameters
- Feature weights
- Threshold values
- Database settings

## 📈 Performance Metrics

- Trust Score Accuracy
- False Positive Rate
- Processing Speed
- Memory Usage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details 