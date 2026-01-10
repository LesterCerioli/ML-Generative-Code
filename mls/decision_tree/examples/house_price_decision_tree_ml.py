import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# Configure visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================================================================
# EXAMPLE 1: CLASSIFICATION WITH DECISION TREE
# ====================================================================

def classification_example():
    """
    Decision tree classification example
    Using Iris dataset (flower species classification)
    """
    print("=" * 60)
    print("EXAMPLE 1: CLASSIFICATION WITH DECISION TREE")
    print("=" * 60)
    
    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"\n📊 Dataset: Iris")
    print(f"📈 Number of samples: {X.shape[0]}")
    print(f"🔢 Number of features: {X.shape[1]}")
    print(f"🎯 Classes: {target_names}")
    
    # 2. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📋 Dataset split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # 3. Create and train model
    tree_clf = DecisionTreeClassifier(
        max_depth=3,           # Maximum tree depth
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=2,    # Minimum samples in a leaf
        random_state=42
    )
    
    tree_clf.fit(X_train, y_train)
    
    print(f"\n🌳 Tree parameters:")
    print(f"  Maximum depth: {tree_clf.get_depth()}")
    print(f"  Number of leaves: {tree_clf.get_n_leaves()}")
    
    # 4. Make predictions
    y_pred_train = tree_clf.predict(X_train)
    y_pred_test = tree_clf.predict(X_test)
    
    # 5. Evaluate model
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\n📊 Performance metrics:")
    print(f"  Training accuracy: {accuracy_train:.2%}")
    print(f"  Testing accuracy:  {accuracy_test:.2%}")
    
    print("\n📋 Classification Report (test):")
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    
    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title('Confusion Matrix - Iris Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Visualize decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, 
              feature_names=feature_names, 
              class_names=target_names,
              filled=True, 
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree - Iris Classification")
    plt.tight_layout()
    plt.savefig('decision_tree_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Feature importance
    importance = tree_clf.feature_importances_
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_names, importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance in Decision Tree')
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tree_clf, X_test, y_test

# ====================================================================
# EXAMPLE 2: REGRESSION WITH DECISION TREE
# ====================================================================

def regression_example():
    """
    Decision tree regression example
    Predicting continuous values (Diabetes dataset)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: REGRESSION WITH DECISION TREE")
    print("=" * 60)
    
    # 1. Load regression dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    print(f"\n📊 Dataset: Diabetes")
    print(f"📈 Number of samples: {X.shape[0]}")
    print(f"🔢 Number of features: {X.shape[1]}")
    print(f"🎯 Target variable: Disease progression (continuous values)")
    
    # 2. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Create and train regression model
    tree_reg = DecisionTreeRegressor(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    tree_reg.fit(X_train, y_train)
    
    # 4. Make predictions
    y_pred_train = tree_reg.predict(X_train)
    y_pred_test = tree_reg.predict(X_test)
    
    # 5. Evaluate model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n📊 Regression Metrics:")
    print(f"  MSE (training): {mse_train:.2f}")
    print(f"  MSE (testing):  {mse_test:.2f}")
    print(f"  R² (training):  {r2_train:.2%}")
    print(f"  R² (testing):   {r2_test:.2%}")
    
    # 6. Visualize predictions vs actual values
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted (training)
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values (Training)')
    plt.ylabel('Predicted Values (Training)')
    plt.title(f'R² Training: {r2_train:.2%}')
    
    # Plot 2: Actual vs Predicted (testing)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5, color='orange')
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values (Testing)')
    plt.ylabel('Predicted Values (Testing)')
    plt.title(f'R² Testing: {r2_test:.2%}')
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Visualize part of regression tree
    plt.figure(figsize=(15, 8))
    plot_tree(tree_reg, 
              feature_names=diabetes.feature_names,
              filled=True, 
              rounded=True,
              max_depth=2,  # Show only 2 levels to avoid clutter
              fontsize=10)
    plt.title("Decision Tree - Regression (first 2 levels)")
    plt.tight_layout()
    plt.savefig('decision_tree_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tree_reg, X_test, y_test

# ====================================================================
# EXAMPLE 3: CUSTOM DATASET - CREDIT DECISION
# ====================================================================

def custom_dataset_example():
    """
    Example with custom dataset simulating credit decision
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CREDIT DECISION - CUSTOM DATASET")
    print("=" * 60)
    
    # Create simulated dataset
    np.random.seed(42)
    n_samples = 500
    
    # Features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(1000, 10000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    employment_time = np.random.randint(0, 40, n_samples)
    
    # Create target variable (approved: 1, denied: 0)
    # Based on simple rules
    approved = (
        (credit_score > 650) & 
        (income > 3000) & 
        (employment_time > 2)
    ).astype(int)
    
    # Add some noise
    noise = np.random.rand(n_samples) > 0.85
    approved[noise] = 1 - approved[noise]
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'employment_time': employment_time,
        'approved': approved
    })
    
    print(f"\n📊 Custom Dataset: Credit Decision")
    print(f"📈 Number of samples: {len(df)}")
    print(f"🎯 Class distribution:")
    print(df['approved'].value_counts(normalize=True).apply(lambda x: f'{x:.1%}'))
    
    # Separate features and target
    X = df.drop('approved', axis=1)
    y = df['approved']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Create and train model
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42
    )
    
    tree.fit(X_train, y_train)
    
    # Evaluate
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Denied', 'Approved']))
    
    # Visualize tree rules
    plt.figure(figsize=(15, 8))
    plot_tree(tree, 
              feature_names=X.columns,
              class_names=['Denied', 'Approved'],
              filled=True, 
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree - Credit Approval")
    plt.tight_layout()
    plt.savefig('credit_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Extract decision rules
    print("\n📝 Main Decision Rules:")
    print("(Analyze the tree above to see all rules)")
    
    return tree, X_test, y_test

# ====================================================================
# HYPERPARAMETER TUNING FUNCTION
# ====================================================================

def tune_hyperparameters(X, y):
    """
    Hyperparameter tuning example using Grid Search
    """
    from sklearn.model_selection import GridSearchCV
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Define parameters to test
    parameters = {
        'max_depth': [2, 3, 4, 5, 6, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Create base model
    base_tree = DecisionTreeClassifier(random_state=42)
    
    # Grid Search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=parameters,
        cv=5,  # 5-fold cross validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Execute search
    grid_search.fit(X, y)
    
    print(f"\n🎯 Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Best score (cross-validation): {grid_search.best_score_:.2%}")
    
    return grid_search.best_estimator_

# ====================================================================
# EXAMPLE EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("🌳 DECISION TREE EXAMPLES IN PYTHON 🌳")
    print("=" * 60)
    
    try:
        # Execute classification example
        model_clf, X_test_clf, y_test_clf = classification_example()
        
        # Execute regression example
        model_reg, X_test_reg, y_test_reg = regression_example()
        
        # Execute custom dataset example
        model_custom, X_test_custom, y_test_custom = custom_dataset_example()
        
        # Demonstrate prediction with new data
        print("\n" + "=" * 60)
        print("DEMONSTRATION: MAKING PREDICTIONS")
        print("=" * 60)
        
        # New example for classification (Iris)
        new_example = [[5.1, 3.5, 1.4, 0.2]]  # Flower characteristics
        
        if 'model_clf' in locals():
            prediction = model_clf.predict(new_example)
            probability = model_clf.predict_proba(new_example)
            
            print(f"\n🌺 Prediction for new flower:")
            print(f"  Features: {new_example[0]}")
            print(f"  Predicted species: {load_iris().target_names[prediction[0]]}")
            print(f"  Probabilities: {probability[0]}")
        
        print("\n✅ All examples executed successfully!")
        print("\n📁 The following files were saved:")
        print("   - confusion_matrix.png")
        print("   - decision_tree_classification.png")
        print("   - feature_importance.png")
        print("   - regression_results.png")
        print("   - decision_tree_regression.png")
        print("   - credit_decision_tree.png")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()