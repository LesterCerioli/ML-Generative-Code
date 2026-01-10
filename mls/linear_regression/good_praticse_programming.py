import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. BASIC CLASSIFICATION EXAMPLE
# ============================================================================

def simple_classification():
    """
    Simple decision tree classification example
    Perfect for understanding the basics
    """
    print("=" * 70)
    print("1. SIMPLE CLASSIFICATION: LOAN APPROVAL PREDICTION")
    print("=" * 70)
    
    # Create simple dataset
    print("\n📊 Creating simple dataset: Loan Applications")
    np.random.seed(42)
    
    # Features
    age = np.random.randint(18, 70, 200)
    income = np.random.randint(20000, 120000, 200)
    credit_score = np.random.randint(300, 850, 200)
    
    # Target: Loan Approved (1) or Denied (0)
    # Simple business rules
    approved = (
        (income > 50000) & 
        (credit_score > 650) & 
        (age > 25)
    ).astype(int)
    
    # Add some noise
    noise = np.random.random(200) < 0.15
    approved[noise] = 1 - approved[noise]
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'loan_approved': approved
    })
    
    print(f"\nDataset created: {len(df)} loan applications")
    print(f"Approval rate: {df['loan_approved'].mean():.1%}")
    
    # Visualize data
    print("\n📈 Visualizing the data...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Age distribution
    axes[0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Age Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Income vs Credit Score
    colors = ['red' if x == 0 else 'green' for x in df['loan_approved']]
    axes[1].scatter(df['income'], df['credit_score'], c=colors, alpha=0.6, s=30)
    axes[1].set_xlabel('Income ($)')
    axes[1].set_ylabel('Credit Score')
    axes[1].set_title('Approved (Green) vs Denied (Red)')
    axes[1].grid(True, alpha=0.3)
    
    # Approval rate
    approval_rate = df['loan_approved'].value_counts(normalize=True)
    axes[2].bar(['Denied', 'Approved'], approval_rate.values * 100, 
                color=['red', 'green'], alpha=0.7)
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_title('Loan Approval Rate')
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('loan_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prepare data for modeling
    X = df[['age', 'income', 'credit_score']]
    y = df['loan_approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🔧 Data split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Train decision tree
    print("\n🌳 Training Decision Tree Classifier...")
    tree_clf = DecisionTreeClassifier(
        max_depth=3,  # Keep it simple for understanding
        min_samples_leaf=5,
        random_state=42
    )
    
    tree_clf.fit(X_train, y_train)
    
    print(f"Tree depth: {tree_clf.get_depth()}")
    print(f"Number of leaves: {tree_clf.get_n_leaves()}")
    
    # Make predictions
    y_pred = tree_clf.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Denied', 'Approved']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize the tree
    print("\n🖼️ Visualizing the decision tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, 
              feature_names=['Age', 'Income', 'Credit Score'],
              class_names=['Denied', 'Approved'],
              filled=True, 
              rounded=True,
              fontsize=12)
    plt.title("Decision Tree for Loan Approval", fontsize=16)
    plt.tight_layout()
    plt.savefig('loan_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    print("\n⭐ Feature Importance:")
    importance = tree_clf.feature_importances_
    for feature, imp in zip(['Age', 'Income', 'Credit Score'], importance):
        print(f"  {feature}: {imp:.3f}")
    
    # Make sample predictions
    print("\n🎯 Sample Predictions:")
    sample_applicants = pd.DataFrame({
        'age': [22, 35, 50],
        'income': [40000, 75000, 90000],
        'credit_score': [600, 720, 800]
    })
    
    predictions = tree_clf.predict(sample_applicants)
    probabilities = tree_clf.predict_proba(sample_applicants)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "APPROVED" if pred == 1 else "DENIED"
        confidence = prob[pred] * 100
        print(f"  Applicant {i+1}: {status} ({confidence:.1f}% confidence)")
    
    return tree_clf, X_test, y_test

# ============================================================================
# 2. HOUSE PRICE REGRESSION EXAMPLE
# ============================================================================

def house_price_regression():
    """
    House price prediction with decision tree regression
    Simple and educational
    """
    print("\n" + "=" * 70)
    print("2. HOUSE PRICE PREDICTION (REGRESSION)")
    print("=" * 70)
    
    # Create housing dataset
    print("\n🏠 Creating housing dataset...")
    np.random.seed(42)
    n_houses = 300
    
    # Generate features
    size_sqft = np.random.randint(800, 3500, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    bathrooms = np.random.randint(1, 4, n_houses)
    age_years = np.random.randint(0, 50, n_houses)
    location_score = np.random.randint(1, 11, n_houses)  # 1-10 score
    
    # Calculate price based on realistic formula
    base_price = 100000
    price = (
        base_price +
        size_sqft * 180 +        # $180 per sqft
        bedrooms * 25000 +       # $25k per bedroom
        bathrooms * 20000 +      # $20k per bathroom
        location_score * 15000 -  # $15k per location point
        age_years * 3000         # $3k depreciation per year
    )
    
    # Add some noise
    price = price * np.random.normal(1, 0.15, n_houses)
    price = np.round(price / 1000) * 1000
    
    # Create DataFrame
    houses = pd.DataFrame({
        'size_sqft': size_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years,
        'location_score': location_score,
        'price': price
    })
    
    print(f"\nDataset: {len(houses)} houses")
    print(f"Average price: ${houses['price'].mean():,.0f}")
    print(f"Price range: ${houses['price'].min():,.0f} - ${houses['price'].max():,.0f}")
    
    # Visualize data
    print("\n📈 Visualizing housing data...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Price distribution
    axes[0, 0].hist(houses['price'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('House Price Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price vs Size
    axes[0, 1].scatter(houses['size_sqft'], houses['price'], alpha=0.6)
    axes[0, 1].set_xlabel('Size (sqft)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Price vs Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Price vs Bedrooms
    price_by_bedrooms = houses.groupby('bedrooms')['price'].mean()
    axes[0, 2].bar(price_by_bedrooms.index, price_by_bedrooms.values, alpha=0.7)
    axes[0, 2].set_xlabel('Bedrooms')
    axes[0, 2].set_ylabel('Average Price ($)')
    axes[0, 2].set_title('Price vs Bedrooms')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Price vs Age
    axes[1, 0].scatter(houses['age_years'], houses['price'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Price vs Age')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Price vs Location
    price_by_location = houses.groupby('location_score')['price'].mean()
    axes[1, 1].plot(price_by_location.index, price_by_location.values, 'o-', linewidth=2)
    axes[1, 1].set_xlabel('Location Score (1-10)')
    axes[1, 1].set_ylabel('Average Price ($)')
    axes[1, 1].set_title('Price vs Location Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation matrix
    numeric_cols = houses.select_dtypes(include=[np.number]).columns
    corr_matrix = houses[numeric_cols].corr()
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Matrix')
    axes[1, 2].set_xticks(range(len(numeric_cols)))
    axes[1, 2].set_yticks(range(len(numeric_cols)))
    axes[1, 2].set_xticklabels(numeric_cols, rotation=45, ha='right')
    axes[1, 2].set_yticklabels(numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[1, 2])
    plt.tight_layout()
    plt.savefig('house_price_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Prepare data for regression
    X = houses[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_score']]
    y = houses['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n🔧 Data split:")
    print(f"  Training: {X_train.shape[0]} houses")
    print(f"  Testing:  {X_test.shape[0]} houses")
    
    # Train regression tree
    print("\n🌳 Training Decision Tree Regressor...")
    tree_reg = DecisionTreeRegressor(
        max_depth=4,  # Control complexity
        min_samples_leaf=5,
        random_state=42
    )
    
    tree_reg.fit(X_train, y_train)
    
    print(f"Tree depth: {tree_reg.get_depth()}")
    print(f"Number of leaves: {tree_reg.get_n_leaves()}")
    
    # Make predictions
    y_pred = tree_reg.predict(X_test)
    
    # Evaluate regression model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"  Mean Absolute Error (MAE): ${mae:,.0f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.0f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Average error: {mae/y_test.mean():.1%} of average price")
    
    # Visualize predictions
    print("\n🖼️ Visualizing predictions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title('Actual vs Predicted Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residuals ($)')
    axes[1].set_title('Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('house_price_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize tree structure
    print("\n🖼️ Visualizing regression tree (first 3 levels)...")
    plt.figure(figsize=(20, 10))
    plot_tree(tree_reg, 
              feature_names=['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Age (years)', 'Location Score'],
              filled=True, 
              rounded=True,
              max_depth=3,  # Show only first 3 levels
              fontsize=11,
              proportion=True)
    plt.title("House Price Prediction Tree (First 3 Levels)", fontsize=16)
    plt.tight_layout()
    plt.savefig('house_price_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    print("\n⭐ Feature Importance for Price Prediction:")
    importance = tree_reg.feature_importances_
    features = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Location']
    
    for feature, imp in zip(features, importance):
        print(f"  {feature}: {imp:.3f}")
    
    # Make sample predictions
    print("\n🎯 Sample House Price Predictions:")
    sample_houses = pd.DataFrame({
        'size_sqft': [1200, 2200, 1800],
        'bedrooms': [2, 4, 3],
        'bathrooms': [1, 3, 2],
        'age_years': [5, 20, 10],
        'location_score': [3, 9, 7]
    })
    
    predictions = tree_reg.predict(sample_houses)
    
    for i, pred in enumerate(predictions):
        print(f"  House {i+1}: ${pred:,.0f}")
    
    return tree_reg, X_test, y_test

# ============================================================================
# 3. UNDERSTANDING OVERFITTING
# ============================================================================

def understand_overfitting():
    """
    Demonstrate overfitting with decision trees
    Educational example
    """
    print("\n" + "=" * 70)
    print("3. UNDERSTANDING OVERFITTING")
    print("=" * 70)
    
    print("\n🎯 Goal: Show how tree depth affects performance")
    
    # Create dataset with some noise
    np.random.seed(42)
    X = np.random.rand(200, 1) * 10
    y = 2 * X.squeeze() + np.sin(X.squeeze()) * 3 + np.random.randn(200) * 2
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Try different tree depths
    depths = [1, 2, 3, 5, 10, 20]
    train_scores = []
    test_scores = []
    
    print("\n🌳 Training trees with different depths...")
    
    for depth in depths:
        # Create and train tree
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        # Calculate scores
        train_score = tree.score(X_train, y_train)  # R² score
        test_score = tree.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"  Depth {str(depth):2s}: Train R² = {train_score:.3f}, Test R² = {test_score:.3f}")
    
    # Find best depth
    best_depth = depths[np.argmax(test_scores)]
    print(f"\n✅ Best depth: {best_depth} (highest test R²)")
    
    # Visualize overfitting
    print("\n📈 Visualizing overfitting...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, depth in enumerate(depths[:6]):
        # Train tree
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        # Generate predictions
        X_range = np.linspace(0, 10, 300).reshape(-1, 1)
        y_pred = tree.predict(X_range)
        
        # Plot
        axes[idx].scatter(X_train, y_train, alpha=0.5, s=20, label='Training data')
        axes[idx].plot(X_range, y_pred, 'r-', linewidth=2, label=f'Depth {depth} prediction')
        axes[idx].set_xlabel('Feature X')
        axes[idx].set_ylabel('Target y')
        axes[idx].set_title(f'Tree Depth = {depth}\nTest R² = {test_scores[idx]:.3f}')
        axes[idx].legend(loc='upper left', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot performance vs depth
    print("\n📊 Performance vs Tree Depth...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(depths, train_scores, 'o-', linewidth=2, markersize=8, label='Training Score')
    plt.plot(depths, test_scores, 's-', linewidth=2, markersize=8, label='Testing Score')
    plt.axvline(x=best_depth, color='r', linestyle='--', alpha=0.7, label=f'Best Depth ({best_depth})')
    
    plt.xlabel('Tree Depth')
    plt.ylabel('R² Score')
    plt.title('The Bias-Variance Tradeoff: Training vs Testing Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Key Insights:")
    print("  - Shallow trees (depth=1-2): Underfitting (high bias)")
    print("  - Optimal depth (depth=3-5): Good balance")
    print("  - Deep trees (depth=10+): Overfitting (high variance)")
    print("  - Training score keeps improving, but test score decreases after optimal point")
    
    return best_depth

# ============================================================================
# 4. HYPERPARAMETER TUNING EXAMPLE
# ============================================================================

def hyperparameter_tuning():
    """
    Simple hyperparameter tuning example
    """
    print("\n" + "=" * 70)
    print("4. HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("=" * 70)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Using Iris dataset:")
    print(f"  Features: {iris.feature_names}")
    print(f"  Classes: {iris.target_names}")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    print(f"\n🔍 Searching over {len(param_grid['max_depth'])} × {len(param_grid['min_samples_split'])} × "
          f"{len(param_grid['min_samples_leaf'])} × {len(param_grid['criterion'])} = "
          f"{len(param_grid['max_depth'])*len(param_grid['min_samples_split'])*len(param_grid['min_samples_leaf'])*len(param_grid['criterion'])} combinations")
    
    # Create GridSearchCV
    print("\n🎯 Running Grid Search (this may take a moment)...")
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    
    # Evaluate on test set
    best_tree = grid_search.best_estimator_
    test_accuracy = best_tree.score(X_test, y_test)
    
    print(f"📈 Test set accuracy: {test_accuracy:.3f}")
    
    # Compare with default tree
    default_tree = DecisionTreeClassifier(random_state=42)
    default_tree.fit(X_train, y_train)
    default_accuracy = default_tree.score(X_test, y_test)
    
    print(f"\n📊 Comparison with default parameters:")
    print(f"  Default tree accuracy:   {default_accuracy:.3f}")
    print(f"  Tuned tree accuracy:     {test_accuracy:.3f}")
    print(f"  Improvement:            +{100*(test_accuracy - default_accuracy):.1f}%")
    
    # Visualize best tree
    print("\n🖼️ Visualizing best decision tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, 
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True, 
              rounded=True,
              fontsize=12,
              max_depth=3)  # Show only first 3 levels
    
    plt.title(f"Best Decision Tree (Accuracy: {test_accuracy:.3f})", fontsize=16)
    plt.tight_layout()
    plt.savefig('best_tuned_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grid_search

# ============================================================================
# 5. FEATURE IMPORTANCE EXPLANATION
# ============================================================================

def feature_importance_demo():
    """
    Demonstrate and explain feature importance
    """
    print("\n" + "=" * 70)
    print("5. UNDERSTANDING FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Create dataset with known importance
    np.random.seed(42)
    n_samples = 500
    
    # Create features with different importance levels
    feature1 = np.random.randn(n_samples)  # Most important
    feature2 = np.random.randn(n_samples) * 0.5  # Somewhat important
    feature3 = np.random.randn(n_samples) * 0.2  # Less important
    noise1 = np.random.randn(n_samples) * 0.1  # Noise feature 1
    noise2 = np.random.randn(n_samples) * 0.1  # Noise feature 2
    
    # Create target based on features
    target = (
        2.5 * feature1 +
        1.5 * feature2 +
        0.5 * feature3 +
        np.random.randn(n_samples) * 0.3
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'Very_Important_Feature': feature1,
        'Somewhat_Important_Feature': feature2,
        'Less_Important_Feature': feature3,
        'Noise_Feature_1': noise1,
        'Noise_Feature_2': noise2,
        'Target': target
    })
    
    # Prepare data
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train decision tree
    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    
    # Get feature importance
    importance = tree.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Feature Importance Analysis:")
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.3f}")
    
    # Visualize feature importance
    print("\n📈 Visualizing feature importance...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot
    bars = axes[0].barh(importance_df['Feature'], importance_df['Importance'], 
                        color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Feature Importance from Decision Tree')
    axes[0].invert_yaxis()  # Most important at top
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        axes[0].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
    
    # Actual vs predicted weights
    actual_importance = [2.5, 1.5, 0.5, 0.0, 0.0]  # Known from data generation
    estimated_importance = importance
    
    x = np.arange(len(X.columns))
    width = 0.35
    
    axes[1].bar(x - width/2, actual_importance, width, label='Actual Contribution', alpha=0.7)
    axes[1].bar(x + width/2, estimated_importance, width, label='Estimated Importance', alpha=0.7)
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Importance/Contribution')
    axes[1].set_title('Actual vs Estimated Feature Importance')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'F{i+1}' for i in range(len(X.columns))], rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Insights:")
    print("  - Decision trees correctly identify important features")
    print("  - Noise features get very low importance scores")
    print("  - Importance scores sum to 1.0")
    print("  - Can be used for feature selection")
    
    return importance_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all educational examples
    """
    print("🎓 DECISION TREE LEARNING SCRIPT")
    print("=" * 70)
    print("\nThis script demonstrates decision trees for machine learning education.")
    print("Each example is designed to teach a specific concept.\n")
    
    try:
        # Run all educational examples
        print("Starting educational examples...\n")
        
        # Example 1: Basic classification
        print("📚 Example 1: Basic Classification")
        model1, X_test1, y_test1 = simple_classification()
        
        # Example 2: Regression
        print("\n📚 Example 2: Regression")
        model2, X_test2, y_test2 = house_price_regression()
        
        # Example 3: Understanding overfitting
        print("\n📚 Example 3: Overfitting Demonstration")
        best_depth = understand_overfitting()
        
        # Example 4: Hyperparameter tuning
        print("\n📚 Example 4: Hyperparameter Tuning")
        grid_search = hyperparameter_tuning()
        
        # Example 5: Feature importance
        print("\n📚 Example 5: Feature Importance")
        importance_df = feature_importance_demo()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 LEARNING SUMMARY")
        print("=" * 70)
        print("\nYou have learned:")
        print("✓ How decision trees work for classification and regression")
        print("✓ How to interpret tree visualizations")
        print("✓ The importance of controlling tree depth to prevent overfitting")
        print("✓ How to tune hyperparameters using GridSearchCV")
        print("✓ How to interpret feature importance scores")
        print("✓ Practical applications: loan approval and house price prediction")
        
        print("\n📁 Files created:")
        print("  - loan_data_visualization.png")
        print("  - confusion_matrix.png")
        print("  - loan_decision_tree.png")
        print("  - house_price_visualization.png")
        print("  - house_price_predictions.png")
        print("  - house_price_tree.png")
        print("  - overfitting_demonstration.png")
        print("  - bias_variance_tradeoff.png")
        print("  - best_tuned_tree.png")
        print("  - feature_importance_demo.png")
        
        print("\n🔧 Try modifying:")
        print("  1. Change max_depth parameter in examples")
        print("  2. Add more features to the datasets")
        print("  3. Try different random_state values")
        print("  4. Create your own simple dataset")
        
        print("\n🌟 Next Steps:")
        print("  1. Learn about Random Forests (ensemble of trees)")
        print("  2. Explore Gradient Boosting (more advanced)")
        print("  3. Practice on Kaggle competitions")
        print("  4. Read the scikit-learn documentation")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\n💡 Troubleshooting:")
        print("  - Make sure all libraries are installed")
        print("  - Check if you have write permission in current directory")
        print("  - Try running examples one at a time")

# ============================================================================
# QUICK START FOR BEGINNERS
# ============================================================================

def quick_start():
    """
    Super simple example for absolute beginners
    """
    print("🚀 QUICK START: YOUR FIRST DECISION TREE")
    print("=" * 70)
    
    # Simplest possible dataset
    # Predicting if someone will buy a product based on age
    
    print("\n📊 Simple Dataset: Will customers buy our product?")
    print("Based on customer age:")
    
    # Create data
    customer_ages = [[18], [25], [30], [35], [40], [50], [60], [65]]
    bought_product = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = Yes, 0 = No
    
    print(f"\nAges: {[age[0] for age in customer_ages]}")
    print(f"Bought: {bought_product}")
    print("Pattern: Younger customers (under 40) buy the product")
    
    # Create and train tree
    simple_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    simple_tree.fit(customer_ages, bought_product)
    
    # Make prediction
    new_customer_age = [[28]]  # 28 years old
    prediction = simple_tree.predict(new_customer_age)
    probability = simple_tree.predict_proba(new_customer_age)
    
    print(f"\n🎯 Prediction for 28-year-old customer:")
    print(f"  Will buy? {'YES' if prediction[0] == 1 else 'NO'}")
    print(f"  Confidence: {probability[0][prediction[0]]:.1%}")
    
    # Visualize the tree
    plt.figure(figsize=(12, 6))
    plot_tree(simple_tree, 
              feature_names=['Age'],
              class_names=['No', 'Yes'],
              filled=True,
              rounded=True,
              fontsize=12)
    plt.title("Simple Decision Tree: Age → Buy Product?", fontsize=14)
    plt.tight_layout()
    plt.savefig('quick_start_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 The tree learned:")
    print("  If Age <= 37.5: Predict 'Yes' (will buy)")
    print("  If Age > 37.5: Predict 'No' (won't buy)")
    print("\n✅ Congratulations! You've created your first ML model!")

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("Select learning mode:")
    print("1. Full Educational Tutorial (Recommended)")
    print("2. Quick Start (Absolute Beginners)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_start()
    else:
        main()
    
    print("\n" + "=" * 70)
    print("📚 Keep Learning!")
    print("=" * 70)