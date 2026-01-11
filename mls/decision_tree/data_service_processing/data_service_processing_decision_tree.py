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
import warnings
warnings.filterwarnings('ignore')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CREATE SERVICE DATA
# ============================================================================

def create_service_data(n_samples=1000):
    """
    Create simulated service data for ML learning
    """
    print("=" * 60)
    print("CREATING SIMULATED SERVICE DATA")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate service features
    service_hours = np.random.exponential(5, n_samples).clip(0.5, 24)
    service_cost = service_hours * np.random.normal(100, 20, n_samples)
    technician_exp = np.random.uniform(1, 20, n_samples)
    travel_distance = np.random.exponential(10, n_samples)
    
    # Generate categorical features
    service_types = np.random.choice(
        ['Installation', 'Maintenance', 'Repair', 'Inspection', 'Consultation'], 
        n_samples,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    priorities = np.random.choice(
        ['Low', 'Medium', 'High', 'Critical'], 
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Generate target variables
    # 1. Service success (binary classification target)
    success_prob = (
        0.3 * (technician_exp / 20) +
        0.2 * (1 - (service_hours / 24)) +
        0.1 * (1 - (travel_distance / 50)) +
        0.4 * np.random.random(n_samples)
    )
    service_success = (success_prob > 0.5).astype(int)
    
    # 2. Customer satisfaction (1-5 rating)
    satisfaction_base = (
        3.0 +
        0.5 * (technician_exp / 20) +
        0.3 * (1 - service_hours / 24) -
        0.2 * (travel_distance / 50)
    )
    satisfaction = np.clip(np.round(satisfaction_base + np.random.normal(0, 0.5, n_samples)), 1, 5)
    
    # 3. Actual completion time (regression target)
    completion_time = (
        service_hours * 1.2 +  # Base time
        travel_distance * 0.1 +  # Travel factor
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'service_type': service_types,
        'service_hours': service_hours,
        'technician_exp': technician_exp,
        'travel_distance': travel_distance,
        'priority': priorities,
        'service_cost': service_cost,
        'service_success': service_success,
        'satisfaction': satisfaction,
        'completion_time': completion_time
    })
    
    print(f"✅ Created {n_samples} service records")
    print(f"📊 Service type distribution:")
    print(df['service_type'].value_counts())
    print(f"\n📈 Success rate: {df['service_success'].mean():.1%}")
    print(f"⭐ Average satisfaction: {df['satisfaction'].mean():.1f}/5")
    print(f"⏱️ Average completion time: {df['completion_time'].mean():.1f} hours")
    
    return df

# ============================================================================
# EXAMPLE 1: SERVICE SUCCESS CLASSIFICATION
# ============================================================================

def service_success_classification(df):
    """
    Predict service success using decision tree
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: SERVICE SUCCESS PREDICTION")
    print("=" * 60)
    
    # Prepare data
    features = ['service_hours', 'technician_exp', 'travel_distance', 'service_cost']
    
    # Encode categorical features
    df_encoded = df.copy()
    le_type = LabelEncoder()
    le_priority = LabelEncoder()
    
    df_encoded['service_type_encoded'] = le_type.fit_transform(df['service_type'])
    df_encoded['priority_encoded'] = le_priority.fit_transform(df['priority'])
    
    features += ['service_type_encoded', 'priority_encoded']
    
    X = df_encoded[features]
    y = df_encoded['service_success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Create and train model
    tree_clf = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    tree_clf.fit(X_train, y_train)
    
    print(f"\n🌳 Tree details:")
    print(f"  Depth: {tree_clf.get_depth()}")
    print(f"  Leaves: {tree_clf.get_n_leaves()}")
    
    # Make predictions
    y_pred = tree_clf.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Performance metrics:")
    print(f"  Accuracy: {accuracy:.2%}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Failed', 'Success']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Service Success Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('service_success_cm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize tree
    plt.figure(figsize=(20, 12))
    plot_tree(tree_clf,
              feature_names=features,
              class_names=['Failed', 'Success'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree - Service Success Prediction")
    plt.tight_layout()
    plt.savefig('service_success_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    importance = tree_clf.feature_importances_
    feature_names_display = ['Hours', 'Tech Exp', 'Distance', 'Cost', 'Type', 'Priority']
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_names_display, importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance for Service Success')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('service_success_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tree_clf, X_test, y_test

# ============================================================================
# EXAMPLE 2: COMPLETION TIME REGRESSION
# ============================================================================

def completion_time_regression(df):
    """
    Predict service completion time using decision tree regression
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: COMPLETION TIME PREDICTION")
    print("=" * 60)
    
    # Prepare data
    features = ['service_hours', 'technician_exp', 'travel_distance', 'service_cost']
    
    # Encode categorical features
    df_encoded = df.copy()
    le_type = LabelEncoder()
    le_priority = LabelEncoder()
    
    df_encoded['service_type_encoded'] = le_type.fit_transform(df['service_type'])
    df_encoded['priority_encoded'] = le_priority.fit_transform(df['priority'])
    
    features += ['service_type_encoded', 'priority_encoded']
    
    X = df_encoded[features]
    y = df_encoded['completion_time']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train regression model
    tree_reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42
    )
    
    tree_reg.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = tree_reg.predict(X_test_scaled)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Regression Metrics:")
    print(f"  MAE: {mae:.2f} hours")
    print(f"  RMSE: {rmse:.2f} hours")
    print(f"  R² Score: {r2:.3f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Completion Time (hours)')
    plt.ylabel('Predicted Completion Time (hours)')
    plt.title(f'R² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Completion Time (hours)')
    plt.ylabel('Residuals (hours)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('completion_time_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    importance = tree_reg.feature_importances_
    feature_names_display = ['Hours', 'Tech Exp', 'Distance', 'Cost', 'Type', 'Priority']
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_names_display, importance, color='orange')
    plt.xlabel('Importance')
    plt.title('Feature Importance for Completion Time')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('completion_time_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return tree_reg, X_test, y_test

# ============================================================================
# EXAMPLE 3: CUSTOMER SATISFACTION ANALYSIS
# ============================================================================

def customer_satisfaction_analysis(df):
    """
    Analyze factors affecting customer satisfaction
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CUSTOMER SATISFACTION ANALYSIS")
    print("=" * 60)
    
    # Convert satisfaction to binary (High: 4-5, Low: 1-3)
    df['high_satisfaction'] = (df['satisfaction'] >= 4).astype(int)
    
    print(f"\n📊 Satisfaction distribution:")
    print(f"  High satisfaction (4-5): {df['high_satisfaction'].mean():.1%}")
    print(f"  Low satisfaction (1-3): {(1 - df['high_satisfaction'].mean()):.1%}")
    
    # Prepare features
    features = ['service_hours', 'technician_exp', 'travel_distance', 'service_cost', 'completion_time']
    
    df_encoded = df.copy()
    le_type = LabelEncoder()
    le_priority = LabelEncoder()
    
    df_encoded['service_type_encoded'] = le_type.fit_transform(df['service_type'])
    df_encoded['priority_encoded'] = le_priority.fit_transform(df['priority'])
    
    features += ['service_type_encoded', 'priority_encoded']
    
    X = df_encoded[features]
    y = df_encoded['high_satisfaction']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    satisfaction_tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=42
    )
    
    satisfaction_tree.fit(X_train, y_train)
    
    # Evaluate
    y_pred = satisfaction_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Analyze decision rules
    print("\n📝 Key Decision Rules for High Satisfaction:")
    print("(From the tree visualization)")
    
    # Visualize simple tree
    plt.figure(figsize=(15, 8))
    plot_tree(satisfaction_tree,
              feature_names=['Hours', 'Exp', 'Distance', 'Cost', 'Time', 'Type', 'Priority'],
              class_names=['Low Sat', 'High Sat'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree - Customer Satisfaction")
    plt.tight_layout()
    plt.savefig('satisfaction_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("\n📈 Correlation with Satisfaction:")
    numeric_features = ['service_hours', 'technician_exp', 'travel_distance', 
                       'service_cost', 'completion_time']
    
    correlations = df[numeric_features + ['satisfaction']].corr()['satisfaction'].drop('satisfaction')
    
    for feature, corr in correlations.items():
        print(f"  {feature:20s}: {corr:+.3f}")
    
    return satisfaction_tree, X_test, y_test

# ============================================================================
# EXAMPLE 4: HYPERPARAMETER TUNING
# ============================================================================

def tune_service_model(df):
    """
    Optimize decision tree hyperparameters
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Prepare data for success prediction
    features = ['service_hours', 'technician_exp', 'travel_distance', 'service_cost']
    
    df_encoded = df.copy()
    le_type = LabelEncoder()
    le_priority = LabelEncoder()
    
    df_encoded['service_type_encoded'] = le_type.fit_transform(df['service_type'])
    df_encoded['priority_encoded'] = le_priority.fit_transform(df['priority'])
    
    features += ['service_type_encoded', 'priority_encoded']
    
    X = df_encoded[features]
    y = df_encoded['service_success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define parameter grid
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'criterion': ['gini', 'entropy']
    }
    
    # Create base model
    base_tree = DecisionTreeClassifier(random_state=42)
    
    # Grid Search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    print("\n🔍 Running grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\n📊 Best cross-validation accuracy: {grid_search.best_score_:.2%}")
    
    # Test on holdout set
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    
    print(f"📈 Test set accuracy: {test_accuracy:.2%}")
    
    # Compare with default model
    default_tree = DecisionTreeClassifier(random_state=42)
    default_tree.fit(X_train, y_train)
    default_accuracy = default_tree.score(X_test, y_test)
    
    print(f"\n🔍 Comparison with default model:")
    print(f"  Default model accuracy: {default_accuracy:.2%}")
    print(f"  Tuned model accuracy:   {test_accuracy:.2%}")
    improvement = (test_accuracy - default_accuracy) * 100
    print(f"  Improvement: {improvement:+.1f}%")
    
    return grid_search.best_estimator_

# ============================================================================
# EXAMPLE 5: SERVICE TYPE ANALYSIS
# ============================================================================

def service_type_analysis(df):
    """
    Analyze patterns by service type
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: SERVICE TYPE ANALYSIS")
    print("=" * 60)
    
    # Group by service type
    service_stats = df.groupby('service_type').agg({
        'service_hours': 'mean',
        'service_cost': 'mean',
        'service_success': 'mean',
        'satisfaction': 'mean',
        'completion_time': 'mean'
    }).round(2)
    
    print("\n📊 Service Type Statistics:")
    print(service_stats.to_string())
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Success rate by service type
    success_rates = df.groupby('service_type')['service_success'].mean()
    axes[0, 0].bar(success_rates.index, success_rates.values)
    axes[0, 0].set_title('Success Rate by Service Type')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average satisfaction by service type
    satisfaction_avg = df.groupby('service_type')['satisfaction'].mean()
    axes[0, 1].bar(satisfaction_avg.index, satisfaction_avg.values, color='orange')
    axes[0, 1].set_title('Average Satisfaction by Service Type')
    axes[0, 1].set_ylabel('Satisfaction (1-5)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Average cost by service type
    cost_avg = df.groupby('service_type')['service_cost'].mean()
    axes[1, 0].bar(cost_avg.index, cost_avg.values, color='green')
    axes[1, 0].set_title('Average Cost by Service Type')
    axes[1, 0].set_ylabel('Cost ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Average hours by service type
    hours_avg = df.groupby('service_type')['service_hours'].mean()
    axes[1, 1].bar(hours_avg.index, hours_avg.values, color='purple')
    axes[1, 1].set_title('Average Hours by Service Type')
    axes[1, 1].set_ylabel('Hours')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('service_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return service_stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all service data processing examples
    """
    print("🔧 SERVICE DATA PROCESSING WITH DECISION TREES")
    print("=" * 60)
    print("\nThis script demonstrates machine learning for service data analysis.")
    print("No databases, no complex systems - just pure ML learning.\n")
    
    try:
        # 1. Create service data
        df = create_service_data(n_samples=1000)
        
        # 2. Service success prediction
        print("\n" + "=" * 60)
        print("PHASE 1: CLASSIFICATION - WILL THE SERVICE SUCCEED?")
        print("=" * 60)
        success_model, X_test_success, y_test_success = service_success_classification(df)
        
        # 3. Completion time prediction
        print("\n" + "=" * 60)
        print("PHASE 2: REGRESSION - HOW LONG WILL IT TAKE?")
        print("=" * 60)
        time_model, X_test_time, y_test_time = completion_time_regression(df)
        
        # 4. Customer satisfaction analysis
        print("\n" + "=" * 60)
        print("PHASE 3: ANALYSIS - WHAT AFFECTS SATISFACTION?")
        print("=" * 60)
        sat_model, X_test_sat, y_test_sat = customer_satisfaction_analysis(df)
        
        # 5. Hyperparameter tuning
        print("\n" + "=" * 60)
        print("PHASE 4: OPTIMIZATION - FINDING THE BEST PARAMETERS")
        print("=" * 60)
        tuned_model = tune_service_model(df)
        
        # 6. Service type analysis
        print("\n" + "=" * 60)
        print("PHASE 5: INSIGHTS - UNDERSTANDING SERVICE TYPES")
        print("=" * 60)
        service_stats = service_type_analysis(df)
        
        # 7. Make predictions on new data
        print("\n" + "=" * 60)
        print("DEMONSTRATION: MAKING PREDICTIONS")
        print("=" * 60)
        
        # Create new service example
        new_service = pd.DataFrame({
            'service_hours': [3.5],
            'technician_exp': [8.2],
            'travel_distance': [12.5],
            'service_cost': [420.75],
            'service_type': ['Repair'],
            'priority': ['Medium']
        })
        
        print(f"\n🔮 Predicting for new service:")
        print(f"  Type: {new_service['service_type'].iloc[0]}")
        print(f"  Hours: {new_service['service_hours'].iloc[0]}h")
        print(f"  Tech Experience: {new_service['technician_exp'].iloc[0]} years")
        print(f"  Travel: {new_service['travel_distance'].iloc[0]} km")
        
        # Prepare features for prediction
        le_type = LabelEncoder()
        le_priority = LabelEncoder()
        
        # Fit encoders with training data (simplified for demo)
        all_types = df['service_type'].unique()
        all_priorities = df['priority'].unique()
        le_type.fit(all_types)
        le_priority.fit(all_priorities)
        
        new_service_encoded = new_service.copy()
        new_service_encoded['service_type_encoded'] = le_type.transform(new_service['service_type'])
        new_service_encoded['priority_encoded'] = le_priority.transform(new_service['priority'])
        
        features = ['service_hours', 'technician_exp', 'travel_distance', 
                   'service_cost', 'service_type_encoded', 'priority_encoded']
        X_new = new_service_encoded[features]
        
        # Make prediction
        success_prob = success_model.predict_proba(X_new)[0]
        print(f"\n🎯 Success Prediction:")
        print(f"  Probability of success: {success_prob[1]:.1%}")
        print(f"  Likely outcome: {'SUCCESS' if success_prob[1] > 0.5 else 'FAILURE'}")
        
        print("\n✅ All examples completed successfully!")
        print("\n📁 Files created:")
        print("   - service_success_cm.png (Confusion Matrix)")
        print("   - service_success_tree.png (Decision Tree)")
        print("   - service_success_importance.png (Feature Importance)")
        print("   - completion_time_predictions.png (Regression Results)")
        print("   - completion_time_importance.png (Regression Features)")
        print("   - satisfaction_tree.png (Satisfaction Analysis)")
        print("   - service_type_analysis.png (Service Type Stats)")
        
        print("\n💡 Key Learnings:")
        print("   1. Decision trees can predict service outcomes")
        print("   2. Feature importance shows what matters most")
        print("   3. Hyperparameter tuning improves accuracy")
        print("   4. Different service types have different patterns")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have all required libraries:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn")

# ============================================================================
# QUICK DEMO
# ============================================================================

def quick_demo():
    """
    Quick demonstration of service data ML
    """
    print("🚀 QUICK DEMO: SERVICE DATA ML")
    print("=" * 50)
    
    # Create small dataset
    df = create_service_data(n_samples=200)
    
    # Simple success prediction
    features = ['service_hours', 'technician_exp', 'travel_distance']
    X = df[features]
    y = df['service_success']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    
    accuracy = tree.score(X_test, y_test)
    print(f"\n📊 Simple Model Accuracy: {accuracy:.1%}")
    
    # Show a simple tree
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=features, class_names=['Fail', 'Success'], 
              filled=True, rounded=True, fontsize=10)
    plt.title("Simple Service Success Prediction Tree")
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 The tree learned rules like:")
    print("   IF technician_exp > 10.5 AND service_hours < 4.8")
    print("   THEN predict SUCCESS")

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("Select mode:")
    print("1. Full Tutorial (All examples)")
    print("2. Quick Demo (Simple example)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        quick_demo()
    else:
        main()
    
    print("\n" + "=" * 60)
    print("🎓 EDUCATIONAL PURPOSE ONLY")
    print("=" * 60)
    print("\nThis script is for learning machine learning concepts.")
    print("It uses simulated data to demonstrate decision trees.")
    print("\nConcepts covered:")
    print("  • Data preparation")
    print("  • Classification trees")
    print("  • Regression trees")
    print("  • Feature importance")
    print("  • Hyperparameter tuning")
    print("  • Model evaluation")
    print("=" * 60)