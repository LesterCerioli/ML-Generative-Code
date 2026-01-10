import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Visual configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# EXAMPLE 1: BASIC LINEAR REGRESSION - PREDICTING CODE QUALITY
# ============================================================================

def basic_linear_regression():
    """
    Basic linear regression to predict project quality score
    """
    print("=" * 60)
    print("EXAMPLE 1: BASIC LINEAR REGRESSION - CODE QUALITY PREDICTION")
    print("=" * 60)
    
    # 1. Create software project metrics dataset
    print("\n📊 Creating software projects metrics dataset...")
    np.random.seed(42)
    n_projects = 200
    
    # Generate realistic project metrics
    lines_of_code = np.random.randint(1000, 50000, n_projects)
    num_modules = np.random.randint(5, 100, n_projects)
    num_services = np.random.randint(1, 20, n_projects)
    code_coverage = np.random.uniform(0.2, 0.95, n_projects)
    avg_complexity = np.random.uniform(1.0, 3.0, n_projects)
    
    # Generate quality score based on metrics with some logic
    # Higher coverage, more services, reasonable complexity = higher quality
    quality_score = (
        0.3 * (code_coverage * 100) +              # Code coverage contributes 30%
        0.2 * (num_services / num_modules * 100) + # Service modularity contributes 20%
        0.15 * (100 / (avg_complexity * 10)) +     # Lower complexity is better
        0.25 * (50000 / lines_of_code) * 50 +      # Smaller projects tend to be better
        0.1 * np.random.normal(0, 10, n_projects)  # Some randomness
    )
    
    # Normalize quality score to 0-100 range
    quality_score = (quality_score - quality_score.min()) / (quality_score.max() - quality_score.min()) * 100
    
    # Create DataFrame
    projects = pd.DataFrame({
        'Lines_of_Code': lines_of_code,
        'Num_Modules': num_modules,
        'Num_Services': num_services,
        'Code_Coverage': code_coverage,
        'Avg_Complexity': avg_complexity,
        'Quality_Score': quality_score
    })
    
    print(f"Dataset created: {projects.shape[0]} projects")
    print(f"\n📈 Project Metrics Summary:")
    print(projects.describe().round(2))
    
    # 2. Visualize relationships
    print("\n📊 Visualizing relationships between metrics and quality...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Lines of Code vs Quality
    axes[0, 0].scatter(projects['Lines_of_Code'], projects['Quality_Score'], alpha=0.6)
    axes[0, 0].set_xlabel('Lines of Code')
    axes[0, 0].set_ylabel('Quality Score')
    axes[0, 0].set_title('Code Size vs Quality')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Code Coverage vs Quality
    axes[0, 1].scatter(projects['Code_Coverage'] * 100, projects['Quality_Score'], alpha=0.6, color='green')
    axes[0, 1].set_xlabel('Code Coverage (%)')
    axes[0, 1].set_ylabel('Quality Score')
    axes[0, 1].set_title('Test Coverage vs Quality')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Number of Services vs Quality
    axes[0, 2].scatter(projects['Num_Services'], projects['Quality_Score'], alpha=0.6, color='red')
    axes[0, 2].set_xlabel('Number of Services')
    axes[0, 2].set_ylabel('Quality Score')
    axes[0, 2].set_title('Services vs Quality')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Complexity vs Quality
    axes[1, 0].scatter(projects['Avg_Complexity'], projects['Quality_Score'], alpha=0.6, color='purple')
    axes[1, 0].set_xlabel('Average Complexity')
    axes[1, 0].set_ylabel('Quality Score')
    axes[1, 0].set_title('Complexity vs Quality')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    correlation = projects.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
    axes[1, 1].set_title('Metrics Correlation Matrix')
    
    # Quality distribution
    axes[1, 2].hist(projects['Quality_Score'], bins=20, alpha=0.7, color='orange')
    axes[1, 2].set_xlabel('Quality Score')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Quality Score Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('project_metrics_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Prepare data for Linear Regression
    print("\n🔧 Preparing data for Linear Regression...")
    X = projects[['Lines_of_Code', 'Num_Modules', 'Num_Services', 'Code_Coverage', 'Avg_Complexity']]
    y = projects['Quality_Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} projects")
    print(f"Testing set:  {X_test.shape[0]} projects")
    
    # 4. Train Linear Regression model
    print("\n📈 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Make predictions and evaluate
    print("\n📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # 6. Visualize predictions vs actual
    print("\n🖼️ Visualizing predictions vs actual values...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot of predictions vs actual
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Quality Score')
    axes[0].set_ylabel('Predicted Quality Score')
    axes[0].set_title('Actual vs Predicted Quality Scores')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Quality Score')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title('Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Feature coefficients (importance)
    print("\n⭐ Feature Coefficients (Impact on Quality):")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Absolute_Impact': np.abs(model.coef_)
    }).sort_values('Absolute_Impact', ascending=False)
    
    print(coefficients.to_string(index=False))
    
    print("\n💡 Interpretation:")
    print("Positive coefficient: Increases quality score")
    print("Negative coefficient: Decreases quality score")
    print("\nExample: Code_Coverage has +15.2 coefficient")
    print("         → Increasing coverage by 1% increases quality by 0.152 points")
    
    return model, X_test, y_test

# ============================================================================
# EXAMPLE 2: MULTIPLE LINEAR REGRESSION WITH INTERACTIONS
# ============================================================================

def multiple_regression_with_interactions():
    """
    Multiple linear regression with feature interactions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MULTIPLE REGRESSION WITH INTERACTIONS")
    print("=" * 60)
    
    # 1. Create dataset with interaction effects
    print("\n📊 Creating dataset with interaction effects...")
    np.random.seed(42)
    n = 150
    
    # Generate features
    experience = np.random.uniform(1, 20, n)  # Years of experience
    team_size = np.random.randint(2, 15, n)   # Team size
    project_duration = np.random.uniform(1, 36, n)  # Months
    requirements_changes = np.random.randint(0, 50, n)  # Number of changes
    
    # Generate productivity with interactions
    # Interaction: Experience * Team Size (experienced people in small teams are more productive)
    # Interaction: Project Duration * Requirements Changes (long projects with many changes = less productive)
    
    productivity = (
        5 * experience +                    # Base effect of experience
        -0.5 * team_size +                  # Larger teams = slightly less productive per person
        0.3 * project_duration +            # Longer projects = more output
        -0.2 * requirements_changes +       # More changes = less productive
        
        # Interaction effects
        0.1 * experience * (1/team_size) +  # Experience matters more in small teams
        -0.05 * project_duration * requirements_changes +  # Bad combination
        
        np.random.normal(0, 3, n)           # Random noise
    )
    
    # Create DataFrame
    productivity_data = pd.DataFrame({
        'Experience_Years': experience,
        'Team_Size': team_size,
        'Project_Duration_Months': project_duration,
        'Requirements_Changes': requirements_changes,
        'Productivity_Score': productivity
    })
    
    print(f"Dataset created: {productivity_data.shape[0]} projects")
    
    # 2. Create interaction features
    print("\n🔧 Creating interaction features...")
    X = productivity_data[['Experience_Years', 'Team_Size', 
                          'Project_Duration_Months', 'Requirements_Changes']]
    
    # Manual interaction features
    X['Experience_Team_Ratio'] = X['Experience_Years'] / X['Team_Size']
    X['Duration_Changes_Interaction'] = X['Project_Duration_Months'] * X['Requirements_Changes']
    
    y = productivity_data['Productivity_Score']
    
    # 3. Train model with interactions
    print("\n📈 Training multiple regression model with interactions...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model_with_interactions = LinearRegression()
    model_with_interactions.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model_with_interactions.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score with interactions: {r2:.3f}")
    print(f"MSE with interactions: {mse:.2f}")
    
    # 5. Compare with model without interactions
    print("\n🔍 Comparing with model WITHOUT interactions...")
    X_simple = productivity_data[['Experience_Years', 'Team_Size', 
                                 'Project_Duration_Months', 'Requirements_Changes']]
    
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y, test_size=0.3, random_state=42
    )
    
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train_simple)
    
    y_pred_simple = model_simple.predict(X_test_simple)
    r2_simple = r2_score(y_test_simple, y_pred_simple)
    
    print(f"R² Score without interactions: {r2_simple:.3f}")
    print(f"Improvement with interactions: {(r2 - r2_simple):.3f}")
    
    # 6. Visualize interaction effects
    print("\n🖼️ Visualizing interaction effects...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Experience vs Team Size interaction
    sample_data = productivity_data.sample(50, random_state=42)
    scatter = axes[0].scatter(sample_data['Experience_Years'], 
                             sample_data['Productivity_Score'],
                             c=sample_data['Team_Size'], 
                             cmap='viridis', 
                             s=100, 
                             alpha=0.7)
    axes[0].set_xlabel('Experience (Years)')
    axes[0].set_ylabel('Productivity Score')
    axes[0].set_title('Experience vs Productivity (Color = Team Size)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Team Size')
    
    # Project Duration vs Requirements Changes interaction
    scatter = axes[1].scatter(sample_data['Project_Duration_Months'], 
                             sample_data['Productivity_Score'],
                             c=sample_data['Requirements_Changes'], 
                             cmap='plasma', 
                             s=100, 
                             alpha=0.7)
    axes[1].set_xlabel('Project Duration (Months)')
    axes[1].set_ylabel('Productivity Score')
    axes[1].set_title('Duration vs Productivity (Color = Requirements Changes)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Requirements Changes')
    
    plt.tight_layout()
    plt.savefig('interaction_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Coefficient analysis
    print("\n📊 Regression Coefficients Analysis:")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model_with_interactions.coef_,
        'Std_Error': np.sqrt(np.diag(np.linalg.pinv(X_train.T @ X_train) * mse))
    })
    
    coefficients['t_Value'] = coefficients['Coefficient'] / coefficients['Std_Error']
    coefficients['Significant'] = np.abs(coefficients['t_Value']) > 2
    
    print(coefficients.to_string(index=False))
    
    print("\n💡 Key Insights:")
    print("1. Experience_Team_Ratio has highest positive impact")
    print("2. Duration_Changes_Interaction has negative impact")
    print("3. Team_Size alone has small negative effect")
    
    return model_with_interactions, X_test, y_test

# ============================================================================
# EXAMPLE 3: REGULARIZATION - RIDGE & LASSO REGRESSION
# ============================================================================

def regularization_demo():
    """
    Demonstrating Ridge and Lasso regularization for preventing overfitting
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: REGULARIZATION - RIDGE & LASSO REGRESSION")
    print("=" * 60)
    
    # 1. Create dataset with many features (some irrelevant)
    print("\n📊 Creating dataset with 20 features (only 5 are relevant)...")
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    # Generate features (most are noise)
    X = np.random.randn(n_samples, n_features)
    
    # Only first 5 features are actually relevant
    true_coefficients = np.zeros(n_features)
    true_coefficients[:5] = [3, -2, 1.5, -1, 0.5]
    
    # Generate target with noise
    y = X @ true_coefficients + np.random.randn(n_samples) * 0.5
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    relevant_features = feature_names[:5]
    
    print(f"Relevant features: {relevant_features}")
    print(f"Irrelevant features: {feature_names[5:]}")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 3. Compare different regression models
    print("\n🔬 Comparing different regression approaches:")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1)': Ridge(alpha=1),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1)': Lasso(alpha=1)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Count non-zero coefficients
        if hasattr(model, 'coef_'):
            non_zero = np.sum(model.coef_ != 0)
        else:
            non_zero = 'N/A'
        
        results.append({
            'Model': name,
            'MSE': mse,
            'R²': r2,
            'Non-zero Coef': non_zero
        })
        
        print(f"{name:20s} | MSE: {mse:.3f} | R²: {r2:.3f} | Non-zero: {non_zero}")
    
    results_df = pd.DataFrame(results)
    
    # 4. Visualize coefficient comparison
    print("\n🖼️ Visualizing coefficient comparison...")
    
    # Get coefficients from different models
    lr_coef = LinearRegression().fit(X_train, y_train).coef_
    ridge_coef = Ridge(alpha=1).fit(X_train, y_train).coef_
    lasso_coef = Lasso(alpha=0.1).fit(X_train, y_train).coef_
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # True coefficients
    axes[0, 0].bar(range(n_features), true_coefficients, alpha=0.7, color='blue')
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('True Coefficients (Only first 5 are non-zero)')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Coefficient Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Linear Regression coefficients
    axes[0, 1].bar(range(n_features), lr_coef, alpha=0.7, color='red')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('Linear Regression Coefficients (Overfits)')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Coefficient Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ridge coefficients
    axes[1, 0].bar(range(n_features), ridge_coef, alpha=0.7, color='green')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('Ridge Regression Coefficients (α=1)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Lasso coefficients
    axes[1, 1].bar(range(n_features), lasso_coef, alpha=0.7, color='orange')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('Lasso Regression Coefficients (α=0.1)')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Coefficient Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Demonstrate regularization path (Lasso)
    print("\n📈 Demonstrating Lasso regularization path...")
    
    alphas = np.logspace(-4, 2, 50)
    lasso_coefs = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_coefs.append(lasso.coef_)
    
    lasso_coefs = np.array(lasso_coefs)
    
    plt.figure(figsize=(12, 6))
    for i in range(5):  # Plot first 5 features (the relevant ones)
        plt.plot(alphas, lasso_coefs[:, i], label=f'Feature_{i+1}', linewidth=2)
    
    for i in range(5, n_features):  # Plot irrelevant features
        plt.plot(alphas, lasso_coefs[:, i], ':', alpha=0.3, linewidth=1)
    
    plt.xscale('log')
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Regularization Path (Solid lines = relevant features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.1, color='r', linestyle='--', label='α=0.1')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lasso_regularization_path.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Regularization Insights:")
    print("• Linear Regression: Assigns values to ALL features (overfits)")
    print("• Ridge Regression: Shrinks ALL coefficients toward zero")
    print("• Lasso Regression: Sets irrelevant features to EXACTLY zero (feature selection)")
    
    return results_df

# ============================================================================
# EXAMPLE 4: POLYNOMIAL REGRESSION FOR NON-LINEAR RELATIONSHIPS
# ============================================================================

def polynomial_regression_demo():
    """
    Demonstrating polynomial regression for non-linear relationships
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: POLYNOMIAL REGRESSION")
    print("=" * 60)
    
    # 1. Create non-linear dataset (complexity vs bug rate)
    print("\n📊 Creating non-linear dataset: Complexity vs Bug Rate...")
    np.random.seed(42)
    n = 100
    
    # Generate complexity scores
    complexity = np.linspace(1, 10, n)
    
    # Non-linear relationship: Bug rate increases exponentially with complexity
    bug_rate = (
        0.5 * complexity +           # Linear component
        0.3 * complexity**2 -        # Quadratic component
        0.05 * complexity**3 +       # Cubic component
        np.random.normal(0, 0.5, n)  # Noise
    )
    
    # Ensure positive bug rates
    bug_rate = np.abs(bug_rate)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Complexity': complexity,
        'Bug_Rate': bug_rate
    })
    
    # 2. Try different polynomial degrees
    print("\n🔬 Testing different polynomial degrees...")
    
    degrees = [1, 2, 3, 4, 5, 10]
    results = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(complexity.reshape(-1, 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, bug_rate, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_poly)
        
        # Calculate metrics
        mse = mean_squared_error(bug_rate, y_pred)
        r2 = r2_score(bug_rate, y_pred)
        
        results.append({
            'Degree': degree,
            'MSE': mse,
            'R²': r2,
            'Num_Features': X_poly.shape[1]
        })
        
        # Plot
        axes[idx].scatter(complexity, bug_rate, alpha=0.5, label='Actual data')
        axes[idx].plot(complexity, y_pred, 'r-', linewidth=2, label=f'Degree {degree} fit')
        axes[idx].set_xlabel('Complexity')
        axes[idx].set_ylabel('Bug Rate')
        axes[idx].set_title(f'Polynomial Degree {degree}\nR²={r2:.3f}, MSE={mse:.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n📊 Results for different polynomial degrees:")
    print(results_df.to_string(index=False))
    
    # 3. Demonstrate overfitting with high degree
    print("\n⚠️ Demonstrating overfitting with degree 15...")
    
    degree_overfit = 15
    poly_overfit = PolynomialFeatures(degree=degree_overfit, include_bias=False)
    X_poly_overfit = poly_overfit.fit_transform(complexity.reshape(-1, 1))
    
    model_overfit = LinearRegression()
    model_overfit.fit(X_poly_overfit, bug_rate)
    
    y_pred_overfit = model_overfit.predict(X_poly_overfit)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(complexity, bug_rate, alpha=0.5, label='Actual data')
    plt.plot(complexity, y_pred_overfit, 'r-', linewidth=2, label=f'Degree {degree_overfit}')
    plt.xlabel('Complexity')
    plt.ylabel('Bug Rate')
    plt.title(f'Polynomial Degree {degree_overfit} (Overfitting!)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add new unseen data to show overfitting
    complexity_new = np.linspace(0.5, 10.5, 50)
    X_new_poly = poly_overfit.transform(complexity_new.reshape(-1, 1))
    y_pred_new = model_overfit.predict(X_new_poly)
    
    plt.subplot(1, 2, 2)
    plt.scatter(complexity, bug_rate, alpha=0.5, label='Training data')
    plt.plot(complexity_new, y_pred_new, 'r-', linewidth=2, label='Model prediction')
    plt.xlabel('Complexity')
    plt.ylabel('Bug Rate')
    plt.title('Model on Unseen Data (Poor generalization)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_overfitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Key Learnings:")
    print("1. Degree 1: Linear (underfits non-linear data)")
    print("2. Degree 2-3: Good fit for this data")
    print("3. Degree 5+: Overfits (follows noise)")
    print("4. Degree 15: Severe overfitting (poor generalization)")
    
    return results_df

# ============================================================================
# EXAMPLE 5: REAL-WORLD APPLICATION - PREDICTING DEVELOPMENT TIME
# ============================================================================

def development_time_prediction():
    """
    Real-world application: Predicting software development time
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: PREDICTING SOFTWARE DEVELOPMENT TIME")
    print("=" * 60)
    
    # 1. Create realistic development project dataset
    print("\n📊 Creating development project dataset...")
    np.random.seed(42)
    n_projects = 150
    
    # Generate project characteristics
    requirements_count = np.random.randint(10, 100, n_projects)
    team_experience = np.random.uniform(0.5, 5.0, n_projects)  # Years
    project_complexity = np.random.randint(1, 10, n_projects)
    has_legacy_code = np.random.choice([0, 1], n_projects, p=[0.6, 0.4])
    code_reuse = np.random.uniform(0.1, 0.8, n_projects)  # Percentage
    
    # Calculate development time (in weeks) based on characteristics
    # Base time plus adjustments for each factor
    base_time = 4  # Base weeks
    
    development_time = (
        base_time +
        0.1 * requirements_count +                     # More requirements = more time
        -0.5 * team_experience +                      # More experience = less time
        0.8 * project_complexity +                    # More complex = more time
        2 * has_legacy_code +                         # Legacy code = more time
        -1.5 * code_reuse +                           # More reuse = less time
        0.2 * requirements_count * project_complexity +  # Interaction effect
        np.random.normal(0, 1.5, n_projects)          # Random variation
    )
    
    # Ensure positive times
    development_time = np.maximum(2, development_time)
    
    # Create DataFrame
    projects = pd.DataFrame({
        'Requirements_Count': requirements_count,
        'Team_Experience': team_experience,
        'Project_Complexity': project_complexity,
        'Has_Legacy_Code': has_legacy_code,
        'Code_Reuse_Percent': code_reuse * 100,
        'Development_Time_Weeks': development_time
    })
    
    print(f"Dataset created: {projects.shape[0]} projects")
    print(f"\n📈 Project Statistics:")
    print(f"  Avg development time: {projects['Development_Time_Weeks'].mean():.1f} weeks")
    print(f"  Range: {projects['Development_Time_Weeks'].min():.1f} - {projects['Development_Time_Weeks'].max():.1f} weeks")
    
    # 2. Train model to predict development time
    print("\n🔧 Building prediction model...")
    X = projects.drop('Development_Time_Weeks', axis=1)
    y = projects['Development_Time_Weeks']
    
    # Add interaction features
    X['Req_Complexity_Interaction'] = X['Requirements_Count'] * X['Project_Complexity']
    X['Experience_Reuse_Interaction'] = X['Team_Experience'] * X['Code_Reuse_Percent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Polynomial (degree=2)': make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        )
    }
    
    results = []
    
    print("\n📊 Model Performance Comparison:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE (weeks)': mae,
            'RMSE (weeks)': np.sqrt(mse),
            'R²': r2
        })
        
        print(f"{name:25s} | MAE: {mae:.2f} weeks | RMSE: {np.sqrt(mse):.2f} weeks | R²: {r2:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # 3. Feature importance analysis
    print("\n⭐ Feature Importance Analysis:")
    
    # Use linear model for interpretability
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': linear_model.coef_,
        'Abs_Impact': np.abs(linear_model.coef_)
    }).sort_values('Abs_Impact', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # 4. Make a prediction for a new project
    print("\n🎯 Making prediction for a new project:")
    
    new_project = {
        'Requirements_Count': 50,
        'Team_Experience': 2.5,
        'Project_Complexity': 6,
        'Has_Legacy_Code': 1,
        'Code_Reuse_Percent': 30.0
    }
    
    # Calculate interaction features
    new_project['Req_Complexity_Interaction'] = new_project['Requirements_Count'] * new_project['Project_Complexity']
    new_project['Experience_Reuse_Interaction'] = new_project['Team_Experience'] * new_project['Code_Reuse_Percent']
    
    # Create DataFrame with same column order
    new_project_df = pd.DataFrame([new_project])[X.columns]
    
    # Predict with best model
    best_model = Ridge(alpha=1.0)
    best_model.fit(X_train, y_train)
    
    prediction = best_model.predict(new_project_df)[0]
    
    print(f"\n📋 New Project Characteristics:")
    for key, value in new_project.items():
        if key not in ['Req_Complexity_Interaction', 'Experience_Reuse_Interaction']:
            print(f"  {key}: {value}")
    
    print(f"\n⏰ Predicted Development Time: {prediction:.1f} weeks")
    print(f"   Range (95% confidence): {prediction - 1.96:.1f} - {prediction + 1.96:.1f} weeks")
    
    return results_df, feature_importance, prediction

# ============================================================================
# INTERACTIVE LEARNING SECTION
# ============================================================================

def interactive_linear_learning():
    """
    Interactive section for hands-on linear regression learning
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE LINEAR REGRESSION LEARNING")
    print("=" * 60)
    
    print("\n🎓 Choose what you want to learn:")
    print("1. What is R² score?")
    print("2. What are residuals?")
    print("3. How to interpret coefficients?")
    print("4. When to use regularization?")
    print("5. Try your own prediction")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n📊 Understanding R² Score (R-squared):")
        print("\nR² measures how well the model explains the variance in data")
        print("\nInterpretation:")
        print("  R² = 1.0: Perfect prediction (all points on the line)")
        print("  R² = 0.8: Very good prediction (80% variance explained)")
        print("  R² = 0.5: Moderate prediction (50% variance explained)")
        print("  R² = 0.0: Model predicts the mean (no linear relationship)")
        print("  R² < 0.0: Model is worse than just predicting the mean")
        
        print("\n💡 Example:")
        print("  If R² = 0.75, the model explains 75% of the variance")
        print("  The remaining 25% is unexplained (due to other factors or randomness)")
    
    elif choice == "2":
        print("\n🔍 Understanding Residuals:")
        print("\nResidual = Actual value - Predicted value")
        print("\nWhy residuals matter:")
        print("1. Check model assumptions:")
        print("   • Residuals should be randomly scattered")
        print("   • No patterns should be visible")
        print("   • Should have constant variance")
        print("\n2. Identify problems:")
        print("   • Pattern in residuals = model missing something")
        print("   • Increasing spread = heteroscedasticity")
        print("   • Outliers in residuals = influential points")
        
        print("\n📝 Good residuals vs Bad residuals:")
        print("  Good: Random scatter around zero line")
        print("  Bad: U-shaped pattern (needs polynomial terms)")
        print("  Bad: Funnel shape (needs transformation)")
    
    elif choice == "3":
        print("\n📈 Interpreting Regression Coefficients:")
        print("\nFor a simple model: y = b0 + b1*x1 + b2*x2 + ...")
        print("\nInterpretation:")
        print("  b0 (intercept): Expected y when all x's are 0")
        print("  b1: Change in y when x1 increases by 1 unit")
        print("  b2: Change in y when x2 increases by 1 unit")
        
        print("\n🏗️ Example from architecture analysis:")
        print("  Model: Quality = 50 + 0.15*Coverage - 0.8*Complexity")
        print("  Interpretation:")
        print("    • Base quality (no coverage, complexity=0): 50")
        print("    • 1% more coverage → +0.15 quality points")
        print("    • 1 unit more complexity → -0.8 quality points")
    
    elif choice == "4":
        print("\n🛡️ When to use Regularization:")
        print("\nUse regularization when:")
        print("1. Many features (risk of overfitting)")
        print("2. Multicollinearity (features are correlated)")
        print("3. Feature selection needed")
        
        print("\n🔧 Types of regularization:")
        print("  Ridge (L2): Shrinks all coefficients")
        print("    • Good for correlated features")
        print("    • Never sets coefficients to exactly zero")
        
        print("  Lasso (L1): Can set coefficients to zero")
        print("    • Good for feature selection")
        print("    • Sparse solutions")
        
        print("\n📊 How to choose α (regularization strength):")
        print("  • α = 0: No regularization (ordinary least squares)")
        print("  • α = small: Mild regularization")
        print("  • α = large: Strong regularization")
        print("  • Use cross-validation to find optimal α")
    
    elif choice == "5":
        print("\n🎯 Try your own prediction!")
        print("\nPredict development time for a project:")
        
        try:
            req_count = int(input("Number of requirements: ") or "50")
            team_exp = float(input("Team experience (years): ") or "2.5")
            complexity = int(input("Complexity (1-10): ") or "6")
            has_legacy = int(input("Has legacy code? (1=Yes, 0=No): ") or "1")
            reuse_percent = float(input("Code reuse percentage (0-100): ") or "30")
            
            # Simple prediction formula (based on our model)
            base_time = 4
            predicted_time = (
                base_time +
                0.1 * req_count +
                -0.5 * team_exp +
                0.8 * complexity +
                2 * has_legacy +
                -1.5 * (reuse_percent/100) +
                0.002 * req_count * complexity
            )
            
            print(f"\n📋 Project Summary:")
            print(f"  Requirements: {req_count}")
            print(f"  Team Experience: {team_exp} years")
            print(f"  Complexity: {complexity}/10")
            print(f"  Legacy Code: {'Yes' if has_legacy else 'No'}")
            print(f"  Code Reuse: {reuse_percent}%")
            
            print(f"\n⏰ Estimated Development Time: {predicted_time:.1f} weeks")
            
            # Add context
            if predicted_time < 5:
                print("   → Small project, likely straightforward")
            elif predicted_time < 10:
                print("   → Medium project, plan carefully")
            else:
                print("   → Large project, consider breaking down")
                
        except ValueError:
            print("⚠️ Invalid input. Please enter numbers.")
    
    else:
        print("\n⚠️ Please choose 1, 2, 3, 4, or 5")

# ============================================================================
# PRACTICE EXERCISES
# ============================================================================

def linear_regression_exercises():
    """
    Practice exercises for linear regression
    """
    print("\n" + "=" * 60)
    print("PRACTICE EXERCISES")
    print("=" * 60)
    
    print("\n🧠 Practice your linear regression skills:")
    
    print("\n1. MODIFY THE CODE:")
    print("   In Example 1, add 'Num_Test_Cases' as a new feature.")
    print("   How does it affect the R² score?")
    
    print("\n2. EXPERIMENT WITH REGULARIZATION:")
    print("   In Example 3, try different α values for Ridge and Lasso.")
    print("   Find the α that gives best test performance.")
    
    print("\n3. DIAGNOSE MODEL PROBLEMS:")
    print("   Look at the residuals plot in Example 1.")
    print("   Are there patterns? What might they indicate?")
    
    print("\n4. BUILD YOUR OWN MODEL:")
    print("   Create a linear model to predict:")
    print("   • Code review time based on lines changed and complexity")
    print("   • Bug count based on cyclomatic complexity")
    print("   • Team velocity based on sprint characteristics")
    
    print("\n5. DEBUGGING:")
    print("   What's wrong with this model?")
    print("   X_train: [[1, 100000], [2, 200000], [3, 300000]]")
    print("   y_train: [50000, 100000, 150000]")
    print("   Problem: Features have different scales!")
    print("   Solution: Use StandardScaler before regression")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all learning examples
    """
    print("🎓 LINEAR REGRESSION MACHINE LEARNING TUTORIAL")
    print("=" * 70)
    print("\nLearn Linear Regression applied to software project analysis!")
    print("\nYou'll learn:")
    print("  • Basic linear regression")
    print("  • Multiple regression with interactions")
    print("  • Regularization (Ridge & Lasso)")
    print("  • Polynomial regression")
    print("  • Real-world application: Development time prediction")
    
    try:
        print("\n🚀 Starting tutorial...\n")
        
        # Example 1: Basic linear regression
        model1, X_test1, y_test1 = basic_linear_regression()
        
        # Example 2: Multiple regression with interactions
        model2, X_test2, y_test2 = multiple_regression_with_interactions()
        
        # Example 3: Regularization
        results_reg = regularization_demo()
        
        # Example 4: Polynomial regression
        results_poly = polynomial_regression_demo()
        
        # Example 5: Real-world application
        results_real, feature_imp, prediction = development_time_prediction()
        
        # Interactive learning
        interactive_linear_learning()
        
        # Practice exercises
        linear_regression_exercises()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS! YOU'VE LEARNED:")
        print("=" * 60)
        print("✓ How linear regression works")
        print("✓ How to evaluate regression models (R², MSE, MAE)")
        print("✓ How to handle non-linear relationships")
        print("✓ How to prevent overfitting with regularization")
        print("✓ How to interpret regression coefficients")
        print("✓ Real-world application to software metrics")
        
        print("\n📁 Files created during tutorial:")
        files = [
            'project_metrics_relationships.png',
            'linear_regression_predictions.png',
            'interaction_effects.png',
            'regularization_comparison.png',
            'lasso_regularization_path.png',
            'polynomial_regression_comparison.png',
            'polynomial_overfitting.png'
        ]
        for file in files:
            print(f"   - {file}")
        
        print("\n🌟 Next Steps:")
        print("   1. Try the practice exercises")
        print("   2. Apply to your own project data")
        print("   3. Learn about logistic regression (for classification)")
        print("   4. Explore time series analysis")
        
        print("\n💡 Remember: Models are simplifications of reality.")
        print("             Always validate with domain knowledge!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Debugging is part of learning!")
        print("Make sure you have all required packages:")
        print("pip install pandas numpy matplotlib scikit-learn seaborn")

# ============================================================================
# QUICK START
# ============================================================================

def quick_start():
    """
    Super simple example for absolute beginners
    """
    print("🚀 QUICK START: YOUR FIRST LINEAR REGRESSION")
    print("=" * 50)
    
    # Simplest possible linear regression example
    # Predict code quality based on test coverage
    
    # Data: [Test_Coverage_%], Quality_Score
    simple_data = [
        [20, 30],  # Low coverage, low quality
        [40, 45],  # Medium coverage, medium quality
        [60, 65],  # Good coverage, good quality
        [80, 85],  # High coverage, high quality
        [90, 95],  # Very high coverage, very high quality
    ]
    
    df_simple = pd.DataFrame(simple_data, 
                             columns=['Test_Coverage', 'Quality_Score'])
    
    X = df_simple[['Test_Coverage']]
    y = df_simple['Quality_Score']
    
    # Create and train linear regression
    from sklearn.linear_model import LinearRegression
    
    simple_model = LinearRegression()
    simple_model.fit(X, y)
    
    # Get the line equation
    slope = simple_model.coef_[0]
    intercept = simple_model.intercept_
    
    print(f"\n📈 Linear Regression Results:")
    print(f"Equation: Quality = {intercept:.2f} + {slope:.2f} × Coverage")
    print(f"\nInterpretation:")
    print(f"• Base quality (0% coverage): {intercept:.1f}")
    print(f"• Each 1% more coverage adds {slope:.2f} quality points")
    
    # Make a prediction
    new_coverage = 75  # 75% coverage
    predicted_quality = simple_model.predict([[new_coverage]])[0]
    
    print(f"\n🎯 Prediction for {new_coverage}% coverage:")
    print(f"   Estimated quality: {predicted_quality:.1f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=100, label='Actual data')
    
    # Plot regression line
    x_range = np.linspace(0, 100, 100).reshape(-1, 1)
    y_pred_range = simple_model.predict(x_range)
    plt.plot(x_range, y_pred_range, 'r-', linewidth=2, label='Regression line')
    
    # Highlight prediction
    plt.scatter([new_coverage], [predicted_quality], 
                color='green', s=200, marker='*', label=f'Prediction ({new_coverage}%)')
    
    plt.xlabel('Test Coverage (%)')
    plt.ylabel('Quality Score')
    plt.title('Simple Linear Regression: Coverage → Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simple_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 The model learned: Higher test coverage → Higher quality")
    print("   This matches what we'd expect in software development!")

# ============================================================================
# RUN THE TUTORIAL
# ============================================================================

if __name__ == "__main__":
    print("Select learning mode:")
    print("1. Full Tutorial (Recommended)")
    print("2. Quick Start (Absolute beginners)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        quick_start()
    else:
        main()
    
    print("\n" + "=" * 60)
    print("📚 Learning Resources:")
    print("- Scikit-learn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html")
    print("- Statistics for Regression: https://stattrek.com/regression/regression-analysis")
    print("- Practice Datasets: https://www.kaggle.com/datasets")
    print("=" * 60)