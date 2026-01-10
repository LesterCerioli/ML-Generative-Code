# 🤖 ML-Powered Microservice Template Generator

## 🚀 Overview
An intelligent Machine Learning system that automatically generates optimized microservice project templates based on your project's specific characteristics. Using regression algorithms, it predicts the best technology stack (FastAPI, Django, Flask) and configuration for your microservice architecture.

**Created by Lester Cerioli** | [GitHub Profile](https://github.com/lestercerioli)

## ✨ Features

### 🤖 Machine Learning Models
- **Linear Regression**: Basic framework selection based on project characteristics
- **Multiple Regression**: Complete template prediction with all components
- **Polynomial Regression**: Captures complex non-linear relationships
- **Regularized Regression** (Ridge/Lasso): Handles multicollinearity and prevents overfitting

### 🏗️ Supported Technologies
| Framework | Database | Cache | Messaging | Use Case |
|-----------|----------|-------|-----------|----------|
| **FastAPI** | PostgreSQL | Redis | Kafka | High-performance APIs |
| **Django** | PostgreSQL | Redis | RabbitMQ | Enterprise applications |
| **Flask** | SQLite | Simple | None | Rapid prototyping |

### 📊 Smart Predictions
The ML models analyze:
- Team size and composition
- Project complexity level
- Expected traffic volume
- Data consistency requirements
- Real-time processing needs
- Deployment frequency

## 📁 Project Structure

```
ml-microservice-generator/
├── 📂 docs/                          # Documentation
├── 📂 examples/                      # Example scripts and templates
├── 📂 generated_projects/            # Auto-generated projects
├── 📂 models/                        # Trained ML models
├── 📂 src/                           # Source code
│   ├── 📂 data/                     # Dataset generation
│   ├── 📂 models/                   # ML model implementations
│   ├── 📂 templates/                # Project templates
│   └── 📂 utils/                    # Helper functions
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Dependencies
├── 📄 train_models.py              # Model training script
├── 📄 generate_template.py         # Interactive generator
└── 📄 quick_start.py              # Quick demo
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/lestercerioli/ml-microservice-generator.git
cd ml-microservice-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
fastapi>=0.104.0  # Optional: for testing generated projects
```

## 🚀 Quick Start

### 1. Generate Your First Template
```bash
python quick_start.py
```

### 2. Interactive Mode
```bash
python generate_template.py
```

### 3. Train Custom Models
```bash
python train_models.py --samples 1000 --test-size 0.2
```

## 🎯 Usage Examples

### Basic Usage
```python
from src.models.template_regression import TemplateRegressionModel

# Initialize and train model
model = TemplateRegressionModel()
model.train()

# Predict template for your project
features = [5, 3, 4, 2, 1, 3]  # [team_size, complexity, traffic, data_needs, real_time, deployment]
predictions, template = model.predict_template(features)

print(f"Recommended Framework: {template['framework']}")
print(f"Database: {template['database']}")
print(f"Cache: {template['cache']}")
```

### Command Line Interface
```bash
# Generate template with specific characteristics
python generate.py --team-size 8 --complexity 4 --traffic 5

# Save template to custom directory
python generate.py --output-dir ./my_project --name "user-service"

# Use custom dataset
python generate.py --dataset ./data/custom_patterns.csv
```

## 📊 How It Works

### 1. Data Generation
The system creates synthetic datasets representing microservice patterns:
```python
# Sample training data
[
    [team_size, complexity, traffic, data_needs, real_time, deployment_freq],
    [3, 2, 3, 2, 1, 2],  # → fastapi_simple
    [5, 4, 5, 3, 4, 3],  # → fastapi_performance
    [8, 5, 4, 5, 2, 1],  # → django_modular
    # ... more patterns
]
```

### 2. ML Training Process
```python
# Linear regression for each component
api_model.fit(X_train, y_api)         # Predicts framework
db_model.fit(X_train, y_db)           # Predicts database
cache_model.fit(X_train, y_cache)     # Predicts caching strategy
msg_model.fit(X_train, y_msg)         # Predicts messaging system
```

### 3. Template Generation
Based on predicted scores, the system selects:
- **Framework**: FastAPI (≥2.3), Django (≥1.7), Flask (otherwise)
- **Database**: PostgreSQL (≥2.3), MongoDB (≥1.7), SQLite (otherwise)
- **Cache**: Redis Cluster (≥2.5), Redis (≥1.8), Simple (≥1.2), None
- **Messaging**: Kafka (≥2.5), RabbitMQ (≥1.8), Celery (≥1.2), None

## 📈 Model Performance

| Model | R² Score | RMSE | Best For |
|-------|----------|------|----------|
| Linear Regression | 0.82 | 0.45 | Basic predictions |
| Polynomial Regression | 0.89 | 0.38 | Complex patterns |
| Ridge Regression | 0.85 | 0.42 | Correlated features |
| Lasso Regression | 0.83 | 0.44 | Feature selection |

## 🏗️ Generated Project Structure

Each generated template includes:

### FastAPI Template
```
project_name/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   └── services/
├── alembic/
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py
└── README.md
```

### Django Template
```
project_name/
├── config/
├── apps/
├── static/
├── media/
├── templates/
├── manage.py
├── requirements.txt
└── Dockerfile
```

### Flask Template
```
project_name/
├── app/
│   ├── static/
│   └── templates/
├── migrations/
├── app.py
└── requirements.txt
```

## 🔧 Configuration Options

### Model Parameters
```yaml
# config/model_config.yaml
linear_regression:
  fit_intercept: true
  normalize: false
  
ridge_regression:
  alpha: 1.0
  solver: 'auto'
  
lasso_regression:
  alpha: 0.1
  selection: 'cyclic'

polynomial_regression:
  degree: 2
  interaction_only: false
```

### Template Thresholds
```python
thresholds = {
    'api': {
        'fastapi': 2.3,
        'django': 1.7,
        'flask': 1.0
    },
    'database': {
        'postgresql': 2.3,
        'mongodb': 1.7,
        'sqlite': 1.0
    }
    # ... more thresholds
}
```

## 📚 Learning Resources

### Machine Learning Concepts
1. **Linear Regression**: Understanding coefficients and R²
2. **Polynomial Features**: Capturing non-linear relationships
3. **Regularization**: Preventing overfitting
4. **Feature Importance**: Identifying key decision factors

### Microservice Patterns
1. **Framework Selection**: When to use FastAPI vs Django vs Flask
2. **Database Choice**: SQL vs NoSQL considerations
3. **Caching Strategies**: Redis, in-memory, or no cache
4. **Messaging Systems**: Event-driven architectures

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_models.py

# Test with coverage
pytest --cov=src tests/
```

### Test Examples
```python
# Test model predictions
def test_template_prediction():
    model = TemplateRegressionModel()
    features = [5, 3, 4, 2, 1, 3]
    predictions, template = model.predict_template(features)
    
    assert template['framework'] in ['fastapi', 'django', 'flask']
    assert template['database'] in ['postgresql', 'mongodb', 'sqlite']
    assert predictions['api']['score'] >= 0
```

## 📊 Results Visualization

The project generates several visualization files:

1. **api_regression_analysis.png**: Linear regression results
2. **template_regression_comparison.png**: Model performance comparison
3. **polynomial_regression_surfaces.png**: 3D visualization of predictions
4. **regularized_regression_comparison.png**: Regularization effects

## 🤝 Contributing

### Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Add new ML algorithms (Random Forest, Neural Networks)
- Support for additional frameworks (Express.js, Spring Boot)
- More template variations
- Improved dataset generation
- Enhanced visualization tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: For excellent ML libraries
- **FastAPI, Django, Flask communities**: For amazing frameworks
- **Microservices.io**: For architectural patterns
- **All contributors**: For making this project better

## 📞 Contact

**Lester Cerioli** - [GitHub](https://github.com/lestercerioli) - lester.cerioli@example.com

Project Link: [https://github.com/lestercerioli/ml-microservice-generator](https://github.com/lestercerioli/ml-microservice-generator)

## 📈 Roadmap

### Phase 1: Core ML Models ✓
- Linear regression for basic predictions
- Multiple output regression
- Basic template generation

### Phase 2: Advanced Features
- [ ] Neural network-based predictions
- [ ] Reinforcement learning for optimization
- [ ] Real-time learning from user feedback
- [ ] Integration with CI/CD pipelines

### Phase 3: Ecosystem Integration
- [ ] VS Code extension
- [ ] CLI tool with npm package
- [ ] Web interface
- [ ] API service

## 🎓 Educational Value

This project serves as an excellent learning resource for:
- **ML Beginners**: Understanding regression algorithms
- **DevOps Engineers**: Automated infrastructure generation
- **Full Stack Developers**: Microservice architecture patterns
- **Data Scientists**: Applied ML in software engineering

---

**⭐ Star this repo if you find it useful!**  
**🐛 Report issues in the GitHub Issues section**  
**💡 Suggest features through Pull Requests**

---


