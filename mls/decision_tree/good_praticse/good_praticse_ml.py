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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_iris, load_diabetes, make_classification
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# EXAMPLE 1: BASIC CLASSIFICATION WITH DECISION TREE
# ============================================================================

def basic_classification_example():
    """
    Basic decision tree classification - perfect for learning
    """
    print("=" * 60)
    print("EXAMPLE 1: BASIC CLASSIFICATION WITH DECISION TREE")
    print("=" * 60)
    
    # 1. Create simple dataset (easy to understand)
    print("\n📊 Creating simple dataset...")
    np.random.seed(42)
    n_samples = 200
    
    # Features: Age and Income
    age = np.random.randint(18, 65, n_samples)
    income = np.random.randint(20000, 100000, n_samples)
    
    # Target: Loan Approval (1 = Approved, 0 = Denied)
    # Simple rule: Approve if (income > 50000 AND age > 25) OR income > 80000
    approved = ((income > 50000) & (age > 25)) | (income > 80000)
    approved = approved.astype(int)
    
    # Add some noise (real data isn't perfect)
    noise = np.random.random(n_samples) < 0.1
    approved[noise] = 1 - approved[noise]
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': age,
        'Annual_Income': income,
        'Loan_Approved': approved
    })
    
    print(f"Dataset created: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Loan Approval Rate: {data['Loan_Approved'].mean():.1%}")
    
    # 2. Visualize the data
    print("\n📈 Visualizing the data...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    colors = ['red' if x == 0 else 'green' for x in data['Loan_Approved']]
    axes[0].scatter(data['Age'], data['Annual_Income'], c=colors, alpha=0.6)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Annual Income ($)')
    axes[0].set_title('Loan Approval: Red=Denied, Green=Approved')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution
    approval_counts = data['Loan_Approved'].value_counts()
    axes[1].bar(['Denied', 'Approved'], approval_counts.values, 
                color=['red', 'green'], alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Loan Approval Distribution')
    for i, count in enumerate(approval_counts.values):
        axes[1].text(i, count + 2, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('loan_approval_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Prepare data for modeling
    print("\n🔧 Preparing data for modeling...")
    X = data[['Age', 'Annual_Income']]
    y = data['Loan_Approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    
    # 4. Train decision tree model
    print("\n🌳 Training Decision Tree Classifier...")
    tree_clf = DecisionTreeClassifier(
        max_depth=3,           # Limit depth for interpretability
        random_state=42
    )
    
    tree_clf.fit(X_train, y_train)
    
    print(f"Tree depth: {tree_clf.get_depth()}")
    print(f"Number of leaves: {tree_clf.get_n_leaves()}")
    
    # 5. Make predictions and evaluate
    print("\n📊 Evaluating model performance...")
    y_pred = tree_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Denied', 'Approved']))
    
    # 6. Visualize the decision tree
    print("\n🖼️ Visualizing the decision tree...")
    plt.figure(figsize=(15, 8))
    plot_tree(tree_clf, 
              feature_names=['Age', 'Annual Income'],
              class_names=['Denied', 'Approved'],
              filled=True, 
              rounded=True,
              fontsize=12)
    plt.title("Decision Tree for Loan Approval", fontsize=16)
    plt.tight_layout()
    plt.savefig('loan_approval_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Explain how the tree works
    print("\n💡 Understanding the Decision Tree:")
    print("- Each node asks a question about a feature")
    print("- Left branch = condition is True")
    print("- Right branch = condition is False")
    print("- Leaf nodes show the prediction and probability")
    print("\nExample: The tree learned rules similar to our original logic!")
    
    # 8. Feature importance
    print("\n⭐ Feature Importance:")
    importance = tree_clf.feature_importances_
    for feature, imp in zip(['Age', 'Annual Income'], importance):
        print(f"  {feature}: {imp:.3f}")
    
    return tree_clf, X_test, y_test

# ============================================================================
# EXAMPLE 2: REGRESSION WITH DECISION TREE
# ============================================================================

def regression_example():
    """
    Decision tree regression - predicting continuous values
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: HOUSE PRICE PREDICTION WITH DECISION TREE")
    print("=" * 60)
    
    # 1. Create housing dataset
    print("\n📊 Creating housing price dataset...")
    np.random.seed(42)
    n_houses = 300
    
    # Features
    size_sqft = np.random.randint(800, 3000, n_houses)
    bedrooms = np.random.randint(1, 5, n_houses)
    bathrooms = np.random.randint(1, 4, n_houses)
    age_years = np.random.randint(0, 50, n_houses)
    
    # Calculate price (with some logic students can understand)
    base_price = 100000  # Base price
    price_per_sqft = 150  # $ per square foot
    bedroom_value = 20000  # Value per bedroom
    bathroom_value = 15000  # Value per bathroom
    age_depreciation = -500  # Depreciation per year
    
    price = (
        base_price +
        size_sqft * price_per_sqft +
        bedrooms * bedroom_value +
        bathrooms * bathroom_value +
        age_years * age_depreciation
    )
    
    # Add some random variation (real-world noise)
    price = price * np.random.normal(1, 0.1, n_houses)
    price = np.round(price / 1000) * 1000  # Round to nearest $1000
    
    # Create DataFrame
    houses = pd.DataFrame({
        'Size_sqft': size_sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_years': age_years,
        'Price': price
    })
    
    print(f"Houses in dataset: {len(houses)}")
    print(f"Average price: ${houses['Price'].mean():,.0f}")
    print(f"Price range: ${houses['Price'].min():,.0f} - ${houses['Price'].max():,.0f}")
    
    # 2. Visualize relationships
    print("\n📈 Visualizing house price relationships...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(houses['Size_sqft'], houses['Price'], alpha=0.6)
    axes[0, 0].set_xlabel('Size (sqft)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Price vs Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(houses['Bedrooms'], houses['Price'], alpha=0.6)
    axes[0, 1].set_xlabel('Number of Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Price vs Bedrooms')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(houses['Age_years'], houses['Price'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Price vs Age')
    axes[1, 0].grid(True, alpha=0.3)
    
    price_by_bathrooms = houses.groupby('Bathrooms')['Price'].mean()
    axes[1, 1].bar(price_by_bathrooms.index, price_by_bathrooms.values, alpha=0.7)
    axes[1, 1].set_xlabel('Number of Bathrooms')
    axes[1, 1].set_ylabel('Average Price ($)')
    axes[1, 1].set_title('Average Price by Bathrooms')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('house_price_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Prepare data for regression
    print("\n🔧 Preparing data for regression...")
    X = houses[['Size_sqft', 'Bedrooms', 'Bathrooms', 'Age_years']]
    y = houses['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 4. Train regression tree
    print("\n🌳 Training Decision Tree Regressor...")
    tree_reg = DecisionTreeRegressor(
        max_depth=4,  # Limit depth to prevent overfitting
        min_samples_leaf=5,
        random_state=42
    )
    
    tree_reg.fit(X_train, y_train)
    
    print(f"Tree depth: {tree_reg.get_depth()}")
    print(f"Number of leaves: {tree_reg.get_n_leaves()}")
    
    # 5. Make predictions and evaluate
    print("\n📊 Evaluating regression model...")
    y_pred = tree_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): ${mae:,.0f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.0f}")
    print(f"R² Score: {r2:.3f}")
    
    # 6. Visualize predictions vs actual
    print("\n🖼️ Visualizing predictions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title('Actual vs Predicted Prices')
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
    
    # 7. Visualize part of the regression tree
    print("\n🖼️ Visualizing regression tree (first 2 levels)...")
    plt.figure(figsize=(18, 8))
    plot_tree(tree_reg, 
              feature_names=['Size', 'Bedrooms', 'Bathrooms', 'Age'],
              filled=True, 
              rounded=True,
              max_depth=2,  # Only show first 2 levels
              fontsize=11,
              proportion=True)
    plt.title("House Price Prediction Tree (First 2 Levels)", fontsize=14)
    plt.tight_layout()
    plt.savefig('house_price_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Feature importance
    print("\n⭐ Feature Importance for Price Prediction:")
    importance = tree_reg.feature_importances_
    for feature, imp in zip(['Size', 'Bedrooms', 'Bathrooms', 'Age'], importance):
        print(f"  {feature}: {imp:.3f}")
    
    return tree_reg, X_test, y_test

# ============================================================================
# EXAMPLE 3: HYPERPARAMETER TUNING (LEARNING OPTIMIZATION)
# ============================================================================

def hyperparameter_tuning_example():
    """
    Learn how hyperparameters affect decision trees
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: UNDERSTANDING HYPERPARAMETERS")
    print("=" * 60)
    
    # 1. Create simple dataset
    print("\n📊 Creating dataset for hyperparameter study...")
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 2. Experiment with different hyperparameters
    print("\n🔬 Experimenting with different tree depths...")
    
    depths = [2, 3, 4, 5, 10, None]  # None means no limit
    train_scores = []
    test_scores = []
    
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Depth {str(depth):4s} | Train: {train_score:.3f} | Test: {test_score:.3f}")
    
    # 3. Visualize overfitting
    print("\n📈 Visualizing bias-variance tradeoff...")
    plt.figure(figsize=(10, 6))
    plt.plot([str(d) for d in depths], train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot([str(d) for d in depths], test_scores, 's-', label='Testing Score', linewidth=2)
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Understanding Overfitting: Training vs Testing Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Optimal Depth?')
    plt.tight_layout()
    plt.savefig('overfitting_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Key Learning:")
    print("- Shallow trees (depth=2): Underfitting (high bias)")
    print("- Moderate depth (depth=3-4): Good balance")
    print("- Very deep trees: Overfitting (high variance)")
    print("- Test score decreases after optimal depth due to overfitting!")
    
    # 4. Try GridSearchCV (automated tuning)
    print("\n🎯 Using GridSearchCV for automated tuning...")
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search (this might take a moment)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found: {grid_search.best_params_}")
    print(f"📊 Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # 5. Compare with default model
    default_tree = DecisionTreeClassifier(random_state=42)
    default_tree.fit(X_train, y_train)
    default_score = default_tree.score(X_test, y_test)
    
    tuned_tree = grid_search.best_estimator_
    tuned_score = tuned_tree.score(X_test, y_test)
    
    print(f"\n📊 Model Comparison:")
    print(f"  Default model test accuracy: {default_score:.3f}")
    print(f"  Tuned model test accuracy:   {tuned_score:.3f}")
    print(f"  Improvement: {100*(tuned_score - default_score):.1f}%")
    
    return grid_search

# ============================================================================
# EXAMPLE 4: REAL-WORLD DATASET - IRIS FLOWERS
# ============================================================================

def iris_dataset_example():
    """
    Classic Iris dataset - perfect for learning
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: CLASSIC IRIS DATASET")
    print("=" * 60)
    
    # 1. Load Iris dataset
    print("\n🌺 Loading Iris flower dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: {target_names}")
    print(f"Samples per class: {np.bincount(y)}")
    
    # 2. Create DataFrame for easier exploration
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = y
    iris_df['species_name'] = [target_names[i] for i in y]
    
    # 3. Visualize the data
    print("\n📈 Visualizing Iris data...")
    
    # Pairplot
    plt.figure(figsize=(12, 10))
    for i, species in enumerate(target_names):
        subset = iris_df[iris_df['species'] == i]
        plt.scatter(subset['sepal length (cm)'], 
                   subset['petal length (cm)'],
                   label=species, alpha=0.7, s=50)
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Iris Flowers: Sepal vs Petal Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('iris_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Train and evaluate decision tree
    print("\n🌳 Training decision tree on Iris data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    iris_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    iris_tree.fit(X_train, y_train)
    
    y_pred = iris_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 5. Visualize the tree
    print("\n🖼️ Visualizing Iris decision tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(iris_tree, 
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=12)
    plt.title("Decision Tree for Iris Flower Classification", fontsize=16)
    plt.tight_layout()
    plt.savefig('iris_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Feature importance
    print("\n⭐ Which features are most important?")
    importance = iris_tree.feature_importances_
    for feature, imp in zip(feature_names, importance):
        print(f"  {feature:20s}: {imp:.3f}")
    
    print("\n💡 Insight: Petal measurements are more important than sepal measurements!")
    
    return iris_tree, X_test, y_test

# ============================================================================
# INTERACTIVE LEARNING SECTION
# ============================================================================

def interactive_learning():
    """
    Interactive section for hands-on learning
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE LEARNING")
    print("=" * 60)
    
    print("\n🎓 Let's explore decision trees together!")
    print("Choose what you want to learn:")
    print("1. How does tree depth affect predictions?")
    print("2. What is feature importance?")
    print("3. How to interpret a decision tree?")
    print("4. See a real prediction example")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n📊 Tree Depth Explanation:")
        print("- Shallow trees (depth=1-2): Simple rules, easy to understand")
        print("- Medium trees (depth=3-5): Good balance of accuracy and interpretability")
        print("- Deep trees (depth>5): Complex rules, might overfit")
        print("\nExample: For loan approval, depth=3 might learn:")
        print("  Level 1: Income > $50,000?")
        print("  Level 2: Age > 25?")
        print("  Level 3: Credit Score > 700?")
    
    elif choice == "2":
        print("\n⭐ Feature Importance:")
        print("- Measures how much each feature contributes to predictions")
        print("- Ranges from 0 (no importance) to 1 (very important)")
        print("- Sum of all importances = 1")
        print("\nIn house price prediction:")
        print("  Size: 0.6 (most important)")
        print("  Location: 0.3")
        print("  Age: 0.1 (least important)")
        print("\nThis helps us understand what the model 'thinks' is important!")
    
    elif choice == "3":
        print("\n🔍 How to Read a Decision Tree:")
        print("1. Start at the top (root node)")
        print("2. Ask the question at each node")
        print("3. Follow the arrow based on answer (True=left, False=right)")
        print("4. Continue until you reach a leaf (end node)")
        print("5. Leaf shows: predicted class and probability")
        print("\nExample path for loan approval:")
        print("  Q1: Income > $50,000? → Yes")
        print("  Q2: Age > 25? → Yes")
        print("  Q3: Debt < $10,000? → No")
        print("  Result: Approved (70% confidence)")
    
    elif choice == "4":
        print("\n🎯 Real Prediction Example:")
        print("Let's predict if a loan should be approved:")
        
        # Create a simple model
        np.random.seed(42)
        X_example = np.array([[25, 30000],  # Young, low income
                              [35, 60000],  # Middle-aged, decent income
                              [45, 90000]]) # Older, high income
        y_example = np.array([0, 1, 1])  # Approved: No, Yes, Yes
        
        example_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
        example_tree.fit(X_example, y_example)
        
        # Make prediction for new person
        new_person = [[30, 55000]]  # 30 years old, $55,000 income
        prediction = example_tree.predict(new_person)
        probability = example_tree.predict_proba(new_person)
        
        print(f"\nNew applicant: 30 years old, $55,000 annual income")
        print(f"Prediction: {'APPROVED' if prediction[0] == 1 else 'DENIED'}")
        print(f"Confidence: {probability[0][prediction[0]]:.1%}")
    
    else:
        print("\n⚠️ Please choose 1, 2, 3, or 4 next time!")
    
    print("\n🌟 Keep learning! Decision trees are just the beginning.")

# ============================================================================
# PRACTICE EXERCISES
# ============================================================================

def practice_exercises():
    """
    Practice exercises for reinforcement learning
    """
    print("\n" + "=" * 60)
    print("PRACTICE EXERCISES")
    print("=" * 60)
    
    print("\n🧠 Test your understanding with these exercises:")
    
    print("\n1. MODIFY THE CODE:")
    print("   In the loan approval example, try changing max_depth from 3 to 5.")
    print("   What happens to the accuracy? Is the tree easier or harder to understand?")
    
    print("\n2. EXPERIMENT WITH DATA:")
    print("   In the house price example, add a new feature 'Distance to City'.")
    print("   How does this affect the predictions and feature importance?")
    
    print("\n3. BUILD YOUR OWN:")
    print("   Create a decision tree to predict:")
    print("   - If it will rain tomorrow (based on temperature, humidity, cloud cover)")
    print("   - Movie ratings (based on genre, director, year)")
    print("   - Student grades (based on study hours, attendance, previous grades)")
    
    print("\n4. DEBUGGING:")
    print("   What's wrong with this code?")
    print("   tree = DecisionTreeClassifier()")
    print("   accuracy = tree.score(X_test, y_test)  # Error!")
    print("   Answer: Need to call tree.fit(X_train, y_train) first!")
    
    print("\n5. THINK CRITICALLY:")
    print("   Q: Why might a decision tree perform poorly on some datasets?")
    print("   A: It might overfit to noise in the training data.")
    print("   Q: When would you NOT use a decision tree?")
    print("   A: When you need very accurate predictions or have very complex relationships.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all learning examples
    """
    print("🎓 DECISION TREE MACHINE LEARNING TUTORIAL")
    print("=" * 60)
    print("\nWelcome! This tutorial will teach you decision trees step by step.")
    print("Each example builds on the previous one. Let's begin!\n")
    
    try:
        # Run all examples
        print("🚀 Starting learning journey...\n")
        
        # Example 1: Basic classification
        model1, X_test1, y_test1 = basic_classification_example()
        
        # Example 2: Regression
        model2, X_test2, y_test2 = regression_example()
        
        # Example 3: Hyperparameter tuning
        grid_search = hyperparameter_tuning_example()
        
        # Example 4: Real dataset
        model4, X_test4, y_test4 = iris_dataset_example()
        
        # Interactive learning
        interactive_learning()
        
        # Practice exercises
        practice_exercises()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS! YOU'VE LEARNED:")
        print("=" * 60)
        print("✓ How decision trees work")
        print("✓ Classification vs Regression trees")
        print("✓ How to prevent overfitting")
        print("✓ How to interpret tree visualizations")
        print("✓ Feature importance analysis")
        print("✓ Hyperparameter tuning basics")
        print("\n📁 Files created during tutorial:")
        print("   - loan_approval_data.png")
        print("   - loan_approval_tree.png")
        print("   - house_price_relationships.png")
        print("   - house_price_predictions.png")
        print("   - house_price_tree.png")
        print("   - overfitting_demo.png")
        print("   - iris_visualization.png")
        print("   - iris_decision_tree.png")
        
        print("\n🌟 Next Steps:")
        print("   1. Try the practice exercises")
        print("   2. Experiment with your own data")
        print("   3. Learn about Random Forests (ensemble of decision trees)")
        print("   4. Explore gradient boosting (more advanced trees)")
        
        print("\n💡 Remember: The best way to learn is to experiment!")
        print("   Change parameters, try new datasets, make mistakes!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Don't worry! Learning involves debugging too.")
        print("Check if you have all required libraries installed.")
        print("Required: pandas, numpy, matplotlib, scikit-learn")

# ============================================================================
# QUICK START FOR BEGINNERS
# ============================================================================

def quick_start():
    """
    Super simple example for absolute beginners
    """
    print("🚀 QUICK START: YOUR FIRST DECISION TREE")
    print("=" * 50)
    
    # Simplest possible example
    # Predict if someone likes soccer based on age
    
    # Data: Age and Soccer Preference (1=Yes, 0=No)
    ages = [[10], [15], [20], [25], [30], [35], [40], [45], [50]]
    likes_soccer = [1, 1, 1, 1, 0, 0, 0, 0, 0]  # Young people like soccer
    
    # Create and train tree
    from sklearn.tree import DecisionTreeClassifier
    
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(ages, likes_soccer)
    
    # Make prediction
    new_person_age = [[22]]  # 22 years old
    prediction = tree.predict(new_person_age)
    
    print(f"\nQuestion: Will a {new_person_age[0][0]}-year-old like soccer?")
    print(f"Prediction: {'YES' if prediction[0] == 1 else 'NO'}")
    
    # Visualize the tiny tree
    plt.figure(figsize=(10, 6))
    plot_tree(tree, 
              feature_names=['Age'],
              class_names=['No', 'Yes'],
              filled=True,
              rounded=True)
    plt.title("Simple Decision Tree: Age → Likes Soccer?", fontsize=14)
    plt.tight_layout()
    plt.savefig('quick_start_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 The tree learned: 'If age < 27.5, predict YES, else NO'")
    print("   This matches our data: younger people like soccer!")

# ============================================================================
# RUN THE TUTORIAL
# ============================================================================

if __name__ == "__main__":
    print("Select learning mode:")
    print("1. Full Tutorial (Recommended for learning)")
    print("2. Quick Start (Absolute beginners)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        quick_start()
    else:
        main()
    
    print("\n" + "=" * 60)
    print("📚 Learning Resources:")
    print("- Scikit-learn documentation: https://scikit-learn.org")
    print("- Decision Tree Theory: https://en.wikipedia.org/wiki/Decision_tree")
    print("- Practice on Kaggle: https://www.kaggle.com")
    print("=" * 60)