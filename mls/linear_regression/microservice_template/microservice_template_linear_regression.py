import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# MICROSERVICE DATASET CREATION
# ============================================================================

class MicroserviceDataset:
    """
    Create synthetic dataset of microservice characteristics
    """
    
    @staticmethod
    def generate_synthetic_data(n_samples=500):
        """
        Generate synthetic dataset for microservice template prediction
        Features: [team_size, complexity, traffic, data_needs, real_time, deployment_freq]
        Targets: [api_score, db_score, cache_score, msg_score]
        """
        np.random.seed(42)
        
        # Generate features
        team_size = np.random.randint(1, 15, n_samples)
        complexity = np.random.randint(1, 6, n_samples)
        traffic = np.random.randint(1, 6, n_samples)
        data_needs = np.random.randint(1, 6, n_samples)
        real_time = np.random.randint(1, 6, n_samples)
        deployment_freq = np.random.randint(1, 6, n_samples)
        
        # Generate target scores based on logical relationships
        # API Framework Score (FastAPI=3, Django=2, Flask=1)
        api_score = (
            0.4 * complexity + 
            0.3 * traffic + 
            0.2 * real_time + 
            0.1 * team_size +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Database Score (PostgreSQL=3, MongoDB=2, SQLite=1)
        db_score = (
            0.5 * data_needs + 
            0.3 * traffic + 
            0.2 * complexity +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Cache Score (Redis Cluster=3, Redis=2, Simple=1, None=0)
        cache_score = (
            0.4 * traffic + 
            0.3 * real_time + 
            0.2 * data_needs + 
            0.1 * complexity +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Messaging Score (Kafka=3, RabbitMQ=2, Celery=1, None=0)
        msg_score = (
            0.5 * real_time + 
            0.3 * deployment_freq + 
            0.2 * traffic +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'team_size': team_size,
            'complexity': complexity,
            'traffic': traffic,
            'data_needs': data_needs,
            'real_time': real_time,
            'deployment_freq': deployment_freq,
            'api_score': api_score,
            'db_score': db_score,
            'cache_score': cache_score,
            'msg_score': msg_score
        })
        
        return data

# ============================================================================
# LINEAR REGRESSION FOR TEMPLATE PREDICTION
# ============================================================================

class TemplateRegressionModel:
    """
    Linear regression models for predicting microservice component scores
    """
    
    def __init__(self):
        self.models = {
            'api': LinearRegression(),
            'database': LinearRegression(),
            'cache': LinearRegression(),
            'messaging': LinearRegression()
        }
        
        self.scaler = StandardScaler()
        self.feature_names = [
            'team_size', 'complexity', 'traffic', 
            'data_needs', 'real_time', 'deployment_freq'
        ]
        
        self.thresholds = {
            'api': {'fastapi': 2.3, 'django': 1.7, 'flask': 1.0},
            'database': {'postgresql': 2.3, 'mongodb': 1.7, 'sqlite': 1.0},
            'cache': {'redis_cluster': 2.5, 'redis': 1.8, 'simple': 1.2, 'none': 0.5},
            'messaging': {'kafka': 2.5, 'rabbitmq': 1.8, 'celery': 1.2, 'none': 0.5}
        }
    
    def train(self, X, y_dict):
        """
        Train regression models for each component
        """
        print("=" * 60)
        print("TRAINING LINEAR REGRESSION MODELS")
        print("=" * 60)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        for component, model in self.models.items():
            y = y_dict[component]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            results[component] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\n📊 {component.upper()} Model Performance:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return results
    
    def predict_template(self, features):
        """
        Predict microservice template based on input features
        """
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        predictions = {}
        
        for component, model in self.models.items():
            score = model.predict(features_scaled)[0]
            predictions[component] = {
                'score': score,
                'selection': self._select_component(component, score)
            }
        
        # Build template configuration
        template = self._build_template(predictions)
        
        return predictions, template
    
    def _select_component(self, component, score):
        """
        Select component based on score thresholds
        """
        thresholds = self.thresholds[component]
        
        for tech, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return tech
        
        return list(thresholds.keys())[-1]  # Return lowest if no match
    
    def _build_template(self, predictions):
        """
        Build complete template configuration
        """
        template = {
            'framework': predictions['api']['selection'],
            'database': predictions['database']['selection'],
            'cache': predictions['cache']['selection'],
            'messaging': predictions['messaging']['selection'],
            'monitoring': 'prometheus' if predictions['cache']['score'] > 2.0 else 'basic',
            'auth': 'jwt' if predictions['api']['selection'] == 'fastapi' else 'session',
            'docs': True
        }
        
        return template

# ============================================================================
# EXAMPLE 1: BASIC LINEAR REGRESSION FOR API SELECTION
# ============================================================================

def basic_regression_example():
    """
    Basic linear regression for API framework selection
    """
    print("=" * 60)
    print("EXAMPLE 1: API FRAMEWORK SELECTION WITH LINEAR REGRESSION")
    print("=" * 60)
    
    # 1. Generate dataset
    print("\n📊 Generating microservice dataset...")
    data = MicroserviceDataset.generate_synthetic_data(n_samples=300)
    
    print(f"Dataset shape: {data.shape}")
    print(f"\n📈 Sample data:")
    print(data.head())
    
    # 2. Prepare data for API framework prediction
    X = data[['team_size', 'complexity', 'traffic', 'real_time']]
    y = data['api_score']
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📦 Data split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples:  {X_test.shape[0]}")
    
    # 4. Train linear regression model
    print("\n📈 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Make predictions
    y_pred = model.predict(X_test)
    
    # 6. Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Coefficients: {model.coef_}")
    print(f"   Intercept: {model.intercept_:.4f}")
    
    # 7. Visualize predictions
    print("\n📊 Visualizing predictions vs actual...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual API Score')
    plt.ylabel('Predicted API Score')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # 8. Show feature importance (coefficients)
    plt.subplot(1, 2, 2)
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)
    
    colors = ['green' if c > 0 else 'red' for c in coefficients['coefficient']]
    plt.barh(coefficients['feature'], coefficients['coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance for API Selection')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('api_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Interpret results
    print("\n💡 INTERPRETING THE MODEL:")
    print("   Higher coefficients mean stronger influence on API score")
    print("\n📊 COEFFICIENT ANALYSIS:")
    for feature, coef in zip(X.columns, model.coef_):
        influence = "positive" if coef > 0 else "negative"
        print(f"   {feature:15s}: {coef:7.4f} ({influence} influence)")
    
    print("\n🎯 PREDICTION EXAMPLE:")
    example_features = [[5, 3, 4, 2]]  # team=5, complexity=3, traffic=4, real_time=2
    prediction = model.predict(example_features)[0]
    
    print(f"   Input: Team=5, Complexity=3, Traffic=4, Real-time=2")
    print(f"   Predicted API Score: {prediction:.2f}")
    
    # Convert score to framework recommendation
    if prediction >= 2.3:
        framework = "FastAPI"
        reason = "High complexity and traffic needs"
    elif prediction >= 1.7:
        framework = "Django"
        reason = "Moderate needs, good for team collaboration"
    else:
        framework = "Flask"
        reason = "Simple requirements, rapid development"
    
    print(f"   Recommended Framework: {framework}")
    print(f"   Reason: {reason}")
    
    return model, X_test, y_test

# ============================================================================
# EXAMPLE 2: MULTIPLE REGRESSION FOR FULL TEMPLATE
# ============================================================================

def multiple_regression_example():
    """
    Multiple linear regression for complete template prediction
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: COMPLETE TEMPLATE PREDICTION WITH MULTIPLE REGRESSION")
    print("=" * 60)
    
    # 1. Generate dataset
    print("\n📊 Generating comprehensive dataset...")
    data = MicroserviceDataset.generate_synthetic_data(n_samples=500)
    
    # 2. Prepare features and targets
    X = data[['team_size', 'complexity', 'traffic', 
              'data_needs', 'real_time', 'deployment_freq']]
    
    y_dict = {
        'api': data['api_score'],
        'database': data['db_score'],
        'cache': data['cache_score'],
        'messaging': data['msg_score']
    }
    
    # 3. Initialize and train template model
    template_model = TemplateRegressionModel()
    results = template_model.train(X, y_dict)
    
    # 4. Visualize all models' performance
    print("\n📊 COMPARING ALL REGRESSION MODELS:")
    
    metrics_df = pd.DataFrame({
        component: [
            results[component]['r2'],
            results[component]['rmse'],
            results[component]['mae'],
            results[component]['cv_mean']
        ]
        for component in results.keys()
    }, index=['R²', 'RMSE', 'MAE', 'CV R²'])
    
    print("\n📈 Performance Metrics:")
    print(metrics_df.T.round(4))
    
    # 5. Visualize comparison
    plt.figure(figsize=(14, 6))
    
    # R² scores comparison
    plt.subplot(1, 2, 1)
    components = list(results.keys())
    r2_scores = [results[c]['r2'] for c in components]
    cv_scores = [results[c]['cv_mean'] for c in components]
    
    x_pos = np.arange(len(components))
    width = 0.35
    
    plt.bar(x_pos - width/2, r2_scores, width, label='Test R²', alpha=0.8)
    plt.bar(x_pos + width/2, cv_scores, width, label='CV R²', alpha=0.8)
    
    plt.xlabel('Component')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x_pos, [c.upper() for c in components])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature importance heatmap
    plt.subplot(1, 2, 2)
    
    importance_matrix = []
    for component in components:
        model = results[component]['model']
        importance_matrix.append(model.coef_)
    
    importance_df = pd.DataFrame(
        importance_matrix,
        index=[c.upper() for c in components],
        columns=template_model.feature_names
    )
    
    sns.heatmap(importance_df.T, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, linewidths=1, cbar_kws={'label': 'Coefficient Value'})
    plt.title('Feature Importance Across Components')
    plt.tight_layout()
    
    plt.savefig('template_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Make sample prediction
    print("\n🎯 SAMPLE PREDICTION:")
    sample_features = [8, 4, 3, 5, 2, 3]  # team=8, complexity=4, etc.
    
    predictions, template = template_model.predict_template(sample_features)
    
    print(f"\n📝 Input Features:")
    features_desc = [
        "Team Size: 8 people",
        "Complexity: High (4/5)",
        "Traffic: Medium (3/5)",
        "Data Needs: Very High (5/5)",
        "Real-time: Low (2/5)",
        "Deployment: Medium (3/5)"
    ]
    
    for desc in features_desc:
        print(f"   {desc}")
    
    print(f"\n📊 Predicted Scores:")
    for component, pred in predictions.items():
        print(f"   {component:10s}: {pred['score']:.2f} → {pred['selection']}")
    
    print(f"\n🏗️  RECOMMENDED TEMPLATE:")
    for key, value in template.items():
        print(f"   {key:12s}: {value}")
    
    return template_model, predictions, template

# ============================================================================
# EXAMPLE 3: POLYNOMIAL REGRESSION FOR COMPLEX PATTERNS
# ============================================================================

def polynomial_regression_example():
    """
    Polynomial regression for capturing non-linear relationships
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: POLYNOMIAL REGRESSION FOR COMPLEX PATTERNS")
    print("=" * 60)
    
    # 1. Generate dataset with non-linear relationships
    print("\n📊 Generating dataset with non-linear patterns...")
    np.random.seed(42)
    n_samples = 400
    
    # Create features with non-linear effects
    complexity = np.random.uniform(1, 5, n_samples)
    traffic = np.random.uniform(1, 5, n_samples)
    
    # Create API score with quadratic relationship
    api_score = (
        2.0 + 
        0.5 * complexity + 
        0.3 * traffic + 
        0.2 * complexity**2 -  # Quadratic term
        0.1 * traffic**2 +     # Quadratic term
        0.3 * complexity * traffic +  # Interaction term
        np.random.normal(0, 0.3, n_samples)
    )
    
    # 2. Prepare data
    X = np.column_stack([complexity, traffic])
    y = api_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Create polynomial regression pipeline
    print("\n🔧 Creating polynomial regression model...")
    
    # Degree 1 (Linear)
    linear_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Degree 2 (Quadratic)
    poly2_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    
    # Degree 3 (Cubic)
    poly3_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    
    # 4. Train and compare models
    models = {
        'Linear (degree=1)': linear_model,
        'Quadratic (degree=2)': poly2_model,
        'Cubic (degree=3)': poly3_model
    }
    
    results = {}
    
    print("\n📊 Training and comparing polynomial degrees...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'n_features': X_train.shape[1] if name == 'Linear' else 
                         model.named_steps['poly'].n_output_features_
        }
        
        print(f"   {name:20s}: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # 5. Visualize the regression surfaces
    print("\n📈 Visualizing regression surfaces...")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Create mesh grid for visualization
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    X1, X2 = np.meshgrid(x1, x2)
    
    for idx, (name, result) in enumerate(results.items(), 1):
        model = result['model']
        
        # Prepare grid for prediction
        grid = np.column_stack([X1.ravel(), X2.ravel()])
        Z = model.predict(grid).reshape(X1.shape)
        
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], y, c='red', alpha=0.6, s=20)
        
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Traffic')
        ax.set_zlabel('API Score')
        ax.set_title(f'{name}\nR² = {result["r2"]:.3f}')
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_surfaces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Show feature expansion for polynomial regression
    print("\n🔍 UNDERSTANDING POLYNOMIAL FEATURES:")
    print("   Quadratic (degree=2) expansion creates:")
    print("   [complexity, traffic] → [complexity, traffic, complexity², traffic², complexity×traffic]")
    print("\n   This captures:")
    print("   - Non-linear relationships")
    print("   - Interaction effects")
    print("   - Diminishing returns")
    
    # 7. Interpretation for microservices
    print("\n💡 APPLICATION TO MICROSERVICES:")
    print("   Quadratic terms help capture:")
    print("   - Diminishing returns of adding more team members")
    print("   - Exponential complexity growth")
    print("   - Synergy between features")
    
    return results

# ============================================================================
# EXAMPLE 4: REGULARIZED REGRESSION (RIDGE & LASSO)
# ============================================================================

def regularized_regression_example():
    """
    Ridge and Lasso regression for handling multicollinearity
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: REGULARIZED REGRESSION FOR ROBUST PREDICTIONS")
    print("=" * 60)
    
    # 1. Generate dataset with correlated features
    print("\n📊 Generating dataset with correlated features...")
    np.random.seed(42)
    n_samples = 500
    
    # Base features
    team_size = np.random.randint(1, 15, n_samples)
    
    # Create correlated features
    complexity = 0.7 * team_size + np.random.normal(0, 2, n_samples)
    traffic = 0.5 * team_size + 0.3 * complexity + np.random.normal(0, 1, n_samples)
    data_needs = 0.6 * complexity + np.random.normal(0, 1.5, n_samples)
    
    # Target: Infrastructure complexity score
    infra_score = (
        0.4 * team_size +
        0.3 * complexity +
        0.3 * traffic +
        0.2 * data_needs +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # 2. Check correlations
    corr_data = pd.DataFrame({
        'team_size': team_size,
        'complexity': complexity,
        'traffic': traffic,
        'data_needs': data_needs,
        'infra_score': infra_score
    })
    
    print("\n📊 Feature Correlations:")
    correlation_matrix = corr_data.corr()
    print(correlation_matrix.round(3))
    
    # 3. Prepare data
    X = corr_data[['team_size', 'complexity', 'traffic', 'data_needs']]
    y = corr_data['infra_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Compare different regression models
    models = {
        'Linear': LinearRegression(),
        'Ridge (alpha=1.0)': Ridge(alpha=1.0),
        'Ridge (alpha=10.0)': Ridge(alpha=10.0),
        'Lasso (alpha=0.1)': Lasso(alpha=0.1),
        'Lasso (alpha=1.0)': Lasso(alpha=1.0)
    }
    
    results = {}
    
    print("\n📊 Training regularized regression models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Get coefficients
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
        else:
            coefficients = np.array([model.intercept_])
        
        results[name] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'coefficients': coefficients,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
        }
        
        print(f"   {name:20s}: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # 5. Visualize coefficient comparison
    print("\n📈 Comparing coefficients across models...")
    
    plt.figure(figsize=(14, 6))
    
    # Coefficient comparison
    plt.subplot(1, 2, 1)
    coefficient_data = []
    
    for name, result in results.items():
        if name != 'Linear' and 'Lasso' not in name:  # Just show key models
            continue
            
        coeffs = result['coefficients']
        coefficient_data.append(pd.Series(coeffs, index=X.columns, name=name))
    
    coeff_df = pd.concat(coefficient_data, axis=1)
    
    x_pos = np.arange(len(X.columns))
    width = 0.25
    
    for idx, (model_name, coeffs) in enumerate(coeff_df.items()):
        offset = width * (idx - len(coeff_df.columns) / 2)
        plt.bar(x_pos + offset, coeffs, width, label=model_name, alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Regularization Effect on Coefficients')
    plt.xticks(x_pos, X.columns)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Performance comparison
    plt.subplot(1, 2, 2)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    colors = ['blue' if 'Linear' in name else 
              'orange' if 'Ridge' in name else 
              'green' for name in model_names]
    
    plt.barh(model_names, r2_scores, color=colors, alpha=0.8)
    plt.xlabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (name, score) in enumerate(zip(model_names, r2_scores)):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('regularized_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Interpretation
    print("\n💡 UNDERSTANDING REGULARIZATION:")
    print("\n   LINEAR REGRESSION:")
    print("   - Can overfit with correlated features")
    print("   - Coefficients may be unstable")
    
    print("\n   RIDGE REGRESSION (L2):")
    print("   - Shrinks coefficients toward zero")
    print("   - Handles multicollinearity well")
    print("   - All features remain in model")
    
    print("\n   LASSO REGRESSION (L1):")
    print("   - Can shrink coefficients to exactly zero")
    print("   - Performs feature selection")
    print("   - Good for sparse solutions")
    
    print("\n🎯 APPLICATION TO MICROSERVICE TEMPLATES:")
    print("   Use Ridge when all features might be relevant")
    print("   Use Lasso to identify most important factors")
    print("   Regularization prevents overfitting to training patterns")
    
    return results

# ============================================================================
# INTERACTIVE TEMPLATE GENERATOR WITH LINEAR REGRESSION
# ============================================================================

def interactive_regression_generator():
    """
    Interactive template generator using linear regression
    """
    print("=" * 60)
    print("🤖 LINEAR REGRESSION TEMPLATE GENERATOR")
    print("=" * 60)
    
    # Generate dataset
    print("\n📊 Loading microservice patterns dataset...")
    data = MicroserviceDataset.generate_synthetic_data(n_samples=500)
    
    # Prepare features and targets
    X = data[['team_size', 'complexity', 'traffic', 
              'data_needs', 'real_time', 'deployment_freq']]
    
    y_dict = {
        'api': data['api_score'],
        'database': data['db_score'],
        'cache': data['cache_score'],
        'messaging': data['msg_score']
    }
    
    # Train model
    template_model = TemplateRegressionModel()
    results = template_model.train(X, y_dict)
    
    print("\n" + "=" * 60)
    print("📝 ENTER YOUR PROJECT CHARACTERISTICS")
    print("=" * 60)
    
    questions = [
        ("Team size (1-15 people): ", 1, 15),
        ("Project complexity (1-5 scale): ", 1, 5),
        ("Expected traffic (1-5 scale): ", 1, 5),
        ("Data consistency needs (1-5 scale): ", 1, 5),
        ("Real-time requirements (1-5 scale): ", 1, 5),
        ("Deployment frequency (1-5 scale): ", 1, 5)
    ]
    
    features = []
    for question, min_val, max_val in questions:
        while True:
            try:
                value = float(input(f"\n{question}"))
                if min_val <= value <= max_val:
                    features.append(value)
                    break
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    print("\n" + "=" * 60)
    print("🧠 ANALYZING WITH LINEAR REGRESSION...")
    print("=" * 60)
    
    # Make prediction
    predictions, template = template_model.predict_template(features)
    
    # Show predictions with confidence intervals
    print("\n📊 REGRESSION PREDICTIONS:")
    print("-" * 40)
    
    for component, pred in predictions.items():
        score = pred['score']
        selection = pred['selection']
        
        # Calculate approximate confidence interval
        ci_lower = score - 0.5
        ci_upper = score + 0.5
        
        print(f"\n{component.upper():12s}:")
        print(f"  Score: {score:.2f} [{ci_lower:.2f} - {ci_upper:.2f}]")
        print(f"  Selection: {selection}")
        
        # Interpretation
        if component == 'api':
            if score >= 2.3:
                print("  Reasoning: High score indicates need for performance")
            elif score >= 1.7:
                print("  Reasoning: Moderate score suggests balanced framework")
            else:
                print("  Reasoning: Low score indicates lightweight needs")
    
    print("\n" + "=" * 60)
    print("🏗️  RECOMMENDED MICROSERVICE TEMPLATE")
    print("=" * 60)
    
    print("\n⚙️  CONFIGURATION:")
    for key, value in template.items():
        print(f"  {key:12s}: {value}")
    
    print("\n📈 REGRESSION INSIGHTS:")
    
    # Show feature contributions
    print("\n🔍 FEATURE CONTRIBUTIONS TO PREDICTIONS:")
    
    # Get coefficients for each model
    for component in ['api', 'database', 'cache', 'messaging']:
        model = template_model.models[component]
        
        # Unscale coefficients
        coefficients = model.coef_
        
        print(f"\n{component.upper():12s} Contributions:")
        for feat, coef, val in zip(template_model.feature_names, coefficients, features):
            contribution = coef * val
            direction = "increases" if contribution > 0 else "decreases"
            print(f"  {feat:20s}: {coef:7.3f} × {val:4.1f} = {contribution:7.3f} ({direction} score)")
    
    # Ask for project generation
    generate = input("\n🚀 Generate project template? (y/n): ").strip().lower()
    
    if generate == 'y':
        project_name = input("Enter project name: ").strip() or "regression-microservice"
        
        print(f"\n📁 Generating '{project_name}'...")
        
        # Create simple project structure
        import os
        project_dir = f"./generated/{project_name}"
        os.makedirs(project_dir, exist_ok=True)
        
        # Create README
        readme_content = f"""# {project_name}

## 📊 Generated using Linear Regression ML Model

### Project Characteristics
- Team Size: {features[0]}
- Complexity: {features[1]}/5
- Traffic: {features[2]}/5
- Data Needs: {features[3]}/5
- Real-time: {features[4]}/5
- Deployment: {features[5]}/5

### Recommended Stack
- Framework: {template['framework']}
- Database: {template['database']}
- Cache: {template['cache']}
- Messaging: {template['messaging']}
- Monitoring: {template['monitoring']}
- Authentication: {template['auth']}

### Regression Scores
"""
        
        for component, pred in predictions.items():
            readme_content += f"- {component}: {pred['score']:.2f} → {pred['selection']}\n"
        
        with open(f"{project_dir}/README.md", "w") as f:
            f.write(readme_content)
        
        print(f"\n✅ Project generated at: {project_dir}")
        print("📄 Check README.md for configuration details")
    
    return template_model, predictions, template

# ============================================================================
# PRACTICE EXERCISES
# ============================================================================

def regression_practice_exercises():
    """
    Practice exercises for linear regression concepts
    """
    print("\n" + "=" * 60)
    print("🧠 LINEAR REGRESSION PRACTICE EXERCISES")
    print("=" * 60)
    
    print("\n1. UNDERSTAND COEFFICIENTS:")
    print("   In the API selection model, complexity has coefficient 0.42.")
    print("   What does this mean?")
    print("   Answer: For each 1-unit increase in complexity, API score increases by 0.42")
    
    print("\n2. INTERPRET R² SCORE:")
    print("   Model has R² = 0.78. What does this tell us?")
    print("   Answer: 78% of variance in API score is explained by the features")
    
    print("\n3. MULTICOLLINEARITY:")
    print("   Team size and complexity are correlated (r=0.7).")
    print("   What problem might this cause?")
    print("   Answer: Unstable coefficients, use Ridge regression")
    
    print("\n4. POLYNOMIAL FEATURES:")
    print("   Why use polynomial regression for microservices?")
    print("   Answer: Captures diminishing returns and interaction effects")
    
    print("\n5. REGULARIZATION TRADEOFF:")
    print("   Ridge vs Lasso: Which to use when?")
    print("   Answer: Ridge for correlated features, Lasso for feature selection")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all linear regression examples
    """
    print("📈 MICROSERVICE TEMPLATE GENERATOR - LINEAR REGRESSION ML")
    print("=" * 60)
    print("\nLearn how linear regression predicts optimal microservice templates!")
    
    print("\n" + "=" * 60)
    print("📚 LEARNING OBJECTIVES:")
    print("=" * 60)
    print("✓ Linear regression fundamentals")
    print("✓ Multiple regression for template prediction")
    print("✓ Polynomial regression for complex patterns")
    print("✓ Regularization with Ridge and Lasso")
    print("✓ Feature importance interpretation")
    
    print("\nSelect learning path:")
    print("1. Full Tutorial (All examples)")
    print("2. Interactive Template Generator")
    print("3. Quick Start - Basic Regression")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    try:
        if choice == "1":
            # Run all examples
            print("\n🚀 Starting comprehensive tutorial...\n")
            
            # Example 1: Basic linear regression
            model1, X_test1, y_test1 = basic_regression_example()
            
            # Example 2: Multiple regression
            model2, predictions2, template2 = multiple_regression_example()
            
            # Example 3: Polynomial regression
            results3 = polynomial_regression_example()
            
            # Example 4: Regularized regression
            results4 = regularized_regression_example()
            
            # Practice exercises
            regression_practice_exercises()
            
        elif choice == "2":
            # Interactive generator
            interactive_regression_generator()
            
        elif choice == "3":
            # Quick start
            basic_regression_example()
            
        else:
            print("⚠️ Please enter 1, 2, or 3")
            main()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 LINEAR REGRESSION KEY LEARNINGS:")
        print("=" * 60)
        print("\n📊 COEFFICIENTS:")
        print("   Show feature impact on predictions")
        print("   Positive = increases recommendation score")
        print("   Negative = decreases recommendation score")
        
        print("\n📈 R² SCORE:")
        print("   Measures model fit (0-1 scale)")
        print("   Higher = better predictions")
        
        print("\n🔧 MODEL TYPES:")
        print("   Linear: Simple relationships")
        print("   Polynomial: Complex patterns")
        print("   Ridge/Lasso: Regularized, robust")
        
        print("\n🎯 APPLICATION:")
        print("   Predict optimal frameworks")
        print("   Balance trade-offs automatically")
        print("   Provide data-driven recommendations")
        
        print("\n📁 FILES GENERATED:")
        print("   - api_regression_analysis.png")
        print("   - template_regression_comparison.png")
        print("   - polynomial_regression_surfaces.png")
        print("   - regularized_regression_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Install required packages: pip install scikit-learn matplotlib seaborn")
        print("2. Ensure Python 3.8+ is installed")
        print("3. Check for sufficient memory")

# ============================================================================
# QUICK REGRESSION DEMO
# ============================================================================

def quick_regression_demo():
    """
    Quick demonstration of linear regression for microservices
    """
    print("🚀 QUICK REGRESSION DEMO")
    print("=" * 50)
    
    # Simple example
    print("\n📊 Simple Linear Regression:")
    print("   Predicting API Framework Score")
    
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([1.2, 1.8, 2.4, 3.0, 3.6])  # API scores
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"\n📈 Model trained:")
    print(f"   Coefficients: {model.coef_}")
    print(f"   Intercept: {model.intercept_:.2f}")
    
    # Predict for new project
    new_project = np.array([[3, 4]])  # complexity=3, traffic=4
    prediction = model.predict(new_project)[0]
    
    print(f"\n🎯 Prediction for new project:")
    print(f"   Input: Complexity=3, Traffic=4")
    print(f"   Predicted API Score: {prediction:.2f}")
    
    if prediction > 2.5:
        print("   Recommendation: FastAPI (high-performance)")
    elif prediction > 1.8:
        print("   Recommendation: Django (balanced)")
    else:
        print("   Recommendation: Flask (lightweight)")
    
    print("\n💡 The regression learned:")
    print(f"   API Score = {model.intercept_:.2f} + {model.coef_[0]:.2f}×Complexity + {model.coef_[1]:.2f}×Traffic")

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 LINEAR REGRESSION FOR MICROSERVICE TEMPLATES")
    print("=" * 60)
    
    print("\n📚 What this does:")
    print("   Uses ML to predict optimal microservice configurations")
    print("   Based on your project's specific characteristics")
    print("   Provides data-driven architecture recommendations")
    
    print("\nSelect mode:")
    print("1. Learn & Practice (Recommended)")
    print("2. Quick Demo")
    print("3. Generate Template Now")
    
    mode = input("\nEnter choice (1-3): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        quick_regression_demo()
    elif mode == "3":
        interactive_regression_generator()
    else:
        print("Starting in default mode...")
        main()
    
    print("\n" + "=" * 60)
    print("🌟 NEXT STEPS:")
    print("=" * 60)
    print("\n1. Experiment with different feature values")
    print("2. Try adding your own project data")
    print("3. Extend with more complex regression models")
    print("4. Integrate with actual project generation")
    
    print("\n📖 RESOURCES:")
    print("- Scikit-learn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html")
    print("- Microservice Patterns: https://microservices.io")
    print("- FastAPI, Django, Flask documentation")