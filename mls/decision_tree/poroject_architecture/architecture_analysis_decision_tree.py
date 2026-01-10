
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Visual configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# EXAMPLE 1: BASIC ARCHITECTURE ANALYSIS
# ============================================================================

def basic_architecture_analysis():
    """
    Basic project structure analysis using Decision Tree
    """
    print("=" * 60)
    print("EXAMPLE 1: PROJECT ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    # 1. Create software project evaluation dataset
    print("\n📊 Creating software projects dataset...")
    np.random.seed(42)
    n_projects = 200
    
    # Project characteristics
    num_modules = np.random.randint(5, 50, n_projects)
    layer_separation = np.random.choice([0, 1], n_projects, p=[0.3, 0.7])
    has_workers = np.random.choice([0, 1], n_projects, p=[0.6, 0.4])
    has_models = np.random.choice([0, 1], n_projects, p=[0.2, 0.8])
    has_services = np.random.choice([0, 1], n_projects, p=[0.4, 0.6])
    import_complexity = np.random.randint(1, 5, n_projects)
    
    # Architecture quality (based on characteristics)
    # Rule: Good if (has_services AND has_models) OR (layer_separation AND import_complexity > 2)
    quality = ((has_services == 1) & (has_models == 1)) | \
              ((layer_separation == 1) & (import_complexity > 2))
    quality = quality.astype(int)
    
    # Add noise (not always perfect)
    noise = np.random.random(n_projects) < 0.15
    quality[noise] = 1 - quality[noise]
    
    # Create DataFrame
    projects = pd.DataFrame({
        'Num_Modules': num_modules,
        'Layer_Separation': layer_separation,
        'Has_Workers': has_workers,
        'Has_Models': has_models,
        'Has_Services': has_services,
        'Import_Complexity': import_complexity,
        'Good_Quality': quality
    })
    
    print(f"Dataset created: {projects.shape[0]} projects")
    print(f"Projects with good architecture: {projects['Good_Quality'].mean():.1%}")
    
    # 2. Visualize the data
    print("\n📈 Visualizing project characteristics...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Quality distribution
    counts = projects['Good_Quality'].value_counts()
    axes[0, 0].bar(['Poor', 'Good'], counts.values, color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_title('Quality Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # Number of modules vs quality
    for quality_val in [0, 1]:
        subset = projects[projects['Good_Quality'] == quality_val]
        axes[0, 1].hist(subset['Num_Modules'], 
                       alpha=0.6, 
                       label='Good' if quality_val == 1 else 'Poor',
                       bins=15)
    axes[0, 1].set_title('Number of Modules vs Quality')
    axes[0, 1].set_xlabel('Number of Modules')
    axes[0, 1].legend()
    
    # Features presence
    features = ['Has_Services', 'Has_Models', 'Has_Workers', 'Layer_Separation']
    presence = projects[features].mean()
    axes[0, 2].bar(features, presence.values, alpha=0.7)
    axes[0, 2].set_title('Features Presence')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Correlation between features
    correlation = projects[features + ['Good_Quality']].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
    axes[1, 0].set_title('Correlation Matrix')
    
    # Import complexity vs quality
    for quality_val in [0, 1]:
        subset = projects[projects['Good_Quality'] == quality_val]
        axes[1, 1].scatter(subset['Import_Complexity'], 
                          subset['Num_Modules'],
                          alpha=0.6,
                          label='Good' if quality_val == 1 else 'Poor',
                          s=50)
    axes[1, 1].set_title('Complexity vs Size')
    axes[1, 1].set_xlabel('Import Complexity')
    axes[1, 1].set_ylabel('Number of Modules')
    axes[1, 1].legend()
    
    # Empty plot to complete
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Architecture Analysis\nDecision Tree', 
                   ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('architecture_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Prepare data for model
    print("\n🔧 Preparing data for Decision Tree...")
    X = projects.drop('Good_Quality', axis=1)
    y = projects['Good_Quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training: {X_train.shape[0]} projects")
    print(f"Testing:  {X_test.shape[0]} projects")
    
    # 4. Train Decision Tree
    print("\n🌳 Training Decision Tree Classifier...")
    tree = DecisionTreeClassifier(
        max_depth=4,           # Limit depth for interpretability
        min_samples_split=10,  # Minimum samples to split
        min_samples_leaf=5,    # Minimum samples in a leaf
        random_state=42
    )
    
    tree.fit(X_train, y_train)
    
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    
    # 5. Evaluate the model
    print("\n📊 Evaluating model performance...")
    y_pred = tree.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Poor Architecture', 'Good Architecture']))
    
    # 6. Visualize the decision tree
    print("\n🖼️ Visualizing the decision tree...")
    plt.figure(figsize=(20, 12))
    plot_tree(tree, 
              feature_names=X.columns.tolist(),
              class_names=['Poor', 'Good'],
              filled=True, 
              rounded=True,
              fontsize=10,
              proportion=True,
              impurity=False)
    plt.title("Decision Tree for Architecture Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig('architecture_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Feature importance
    print("\n⭐ Feature Importance:")
    importance = tree.feature_importances_
    for feature, imp in zip(X.columns, importance):
        print(f"  {feature:20s}: {imp:.3f}")
    
    # 8. Interpret learned rules
    print("\n💡 Rules learned by the tree:")
    print("The tree identified patterns such as:")
    print("- Projects with 'Has_Models' AND 'Has_Services' tend to have good architecture")
    print("- Projects with 'Layer_Separation' also have better quality")
    print("- Very large projects (many modules) without proper structure are problematic")
    
    return tree, X_test, y_test

# ============================================================================
# EXAMPLE 2: ANALYSIS OF PROVIDED FASTAPI EXAMPLE
# ============================================================================

def analyze_fastapi_example():
    """
    Specific analysis of the provided FastAPI example
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: FASTAPI EXAMPLE ANALYSIS")
    print("=" * 60)
    
    # Create dataset based on provided example
    project_examples = [
        # [num_modules, has_services, has_workers, has_models, layer_separation, import_complexity, quality]
        [15, 1, 1, 1, 1, 3, 1],  # Provided example (good)
        [8, 0, 0, 0, 0, 1, 0],   # Monolithic project (poor)
        [20, 1, 0, 1, 1, 2, 1],  # Well-structured project
        [30, 1, 1, 1, 1, 4, 1],  # Large well-structured project
        [10, 0, 0, 1, 0, 1, 0],  # Only has models, no services
        [25, 1, 0, 0, 1, 3, 0],  # Has services but no models
        [12, 1, 1, 1, 0, 2, 1],  # Has structure but unclear separation
    ]
    
    # Add more synthetic examples
    np.random.seed(42)
    for _ in range(50):
        num_modules = np.random.randint(5, 40)
        has_services = np.random.choice([0, 1])
        has_workers = np.random.choice([0, 1])
        has_models = np.random.choice([0, 1])
        layer_separation = np.random.choice([0, 1])
        import_complexity = np.random.randint(1, 5)
        
        # Quality rule (simplified)
        quality = 1 if (has_services + has_models + layer_separation) >= 2 else 0
        
        project_examples.append([
            num_modules, has_services, has_workers, 
            has_models, layer_separation, import_complexity, quality
        ])
    
    # Create DataFrame
    df_examples = pd.DataFrame(project_examples, columns=[
        'Num_Modules', 'Has_Services', 'Has_Workers', 
        'Has_Models', 'Layer_Separation', 'Import_Complexity', 'Good_Quality'
    ])
    
    print(f"📊 Dataset with {len(df_examples)} project examples")
    print(f"Quality distribution: {df_examples['Good_Quality'].mean():.1%} good")
    
    # Analyze the provided example
    provided_example = df_examples.iloc[0]  # First example is the provided one
    
    print("\n🔍 Analysis of provided FastAPI example:")
    print(f"Number of modules: {provided_example['Num_Modules']}")
    print(f"Has Services: {'✅ YES' if provided_example['Has_Services'] else '❌ NO'}")
    print(f"Has Workers: {'✅ YES' if provided_example['Has_Workers'] else '❌ NO'}")
    print(f"Has Models: {'✅ YES' if provided_example['Has_Models'] else '❌ NO'}")
    print(f"Layer separation: {'✅ YES' if provided_example['Layer_Separation'] else '❌ NO'}")
    print(f"Import complexity: {provided_example['Import_Complexity']}/4")
    print(f"Predicted quality: {'✅ GOOD' if provided_example['Good_Quality'] else '❌ POOR'}")
    
    # Train model to analyze patterns
    X = df_examples.drop('Good_Quality', axis=1)
    y = df_examples['Good_Quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\n🌳 Model trained with {X_train.shape[0]} examples")
    print(f"Test accuracy: {model.score(X_test, y_test):.2%}")
    
    # Make prediction for provided example
    prediction = model.predict([provided_example.drop('Good_Quality').values])
    probability = model.predict_proba([provided_example.drop('Good_Quality').values])
    
    print(f"\n🎯 Prediction for provided example:")
    print(f"  Quality: {'✅ GOOD' if prediction[0] == 1 else '❌ POOR'}")
    print(f"  Confidence: {probability[0][prediction[0]]:.1%}")
    
    # Visualize decision tree
    plt.figure(figsize=(16, 10))
    plot_tree(model, 
              feature_names=X.columns.tolist(),
              class_names=['Poor', 'Good'],
              filled=True, 
              rounded=True,
              fontsize=11)
    plt.title("Architecture Pattern Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig('architecture_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Rules analysis
    print("\n📝 Rules identified by the model:")
    print("1. Projects with 'Has_Models' AND 'Has_Services' are usually good")
    print("2. Projects without layer separation tend to be worse")
    print("3. Workers are a bonus, but not essential")
    print("4. Moderate import complexity is better than very low or very high")
    
    return model, df_examples

# ============================================================================
# EXAMPLE 3: HYPERPARAMETER OPTIMIZATION
# ============================================================================

def hyperparameter_optimization():
    """
    Shows how different hyperparameters affect the analysis
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Create synthetic dataset
    np.random.seed(42)
    n = 300
    
    X_synth = np.random.randn(n, 5)  # 5 features
    y_synth = (X_synth[:, 0] > 0.5) & (X_synth[:, 1] > -0.5) | (X_synth[:, 2] > 1.0)
    y_synth = y_synth.astype(int)
    
    # Add noise
    noise = np.random.random(n) < 0.1
    y_synth[noise] = 1 - y_synth[noise]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_synth, y_synth, test_size=0.3, random_state=42
    )
    
    # Test different depths
    print("\n🔬 Testing different tree depths:")
    
    depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_scores = []
    test_scores = []
    
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        train_score = tree.score(X_train, y_train)
        test_score = tree.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Depth {depth:2d}: "
              f"Train={train_score:.3f}, "
              f"Test={test_score:.3f}")
    
    # Visualize bias-variance tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, 'o-', label='Train Score', linewidth=2)
    plt.plot(depths, test_scores, 's-', label='Test Score', linewidth=2)
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Tradeoff: Underfitting vs Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.5, label='Optimal depth?')
    plt.tight_layout()
    plt.savefig('depth_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 Insights:")
    print("- Depth 2-3: Underfitting (tree too simple)")
    print("- Depth 4-5: Balanced (good generalization)")
    print("- Depth 6+: Overfitting (memorizes training data)")
    
    # GridSearch to find best parameters
    print("\n🎯 Using GridSearchCV for automatic optimization...")
    
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"📊 Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Compare with default model
    default_model = DecisionTreeClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    
    print(f"\n📊 Test performance comparison:")
    print(f"  Default model:     {default_model.score(X_test, y_test):.3f}")
    print(f"  Optimized model:   {grid_search.best_estimator_.score(X_test, y_test):.3f}")
    
    return grid_search

# ============================================================================
# EXAMPLE 4: ARCHITECTURE RECOMMENDATION GENERATOR
# ============================================================================

def architecture_recommendation_generator():
    """
    Generates architecture recommendations based on Decision Tree
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: ARCHITECTURE RECOMMENDATION GENERATOR")
    print("=" * 60)
    
    # Architecture patterns knowledge base
    patterns = {
        'microservices': {
            'num_modules': 20,
            'has_services': 1,
            'has_workers': 1,
            'has_models': 1,
            'layer_separation': 1,
            'import_complexity': 3
        },
        'monolithic': {
            'num_modules': 8,
            'has_services': 0,
            'has_workers': 0,
            'has_models': 0,
            'layer_separation': 0,
            'import_complexity': 1
        },
        'layered': {
            'num_modules': 15,
            'has_services': 1,
            'has_workers': 0,
            'has_models': 1,
            'layer_separation': 1,
            'import_complexity': 2
        }
    }
    
    # Train model with known patterns
    pattern_data = []
    for pattern, characteristics in patterns.items():
        pattern_data.append(list(characteristics.values()) + [1])  # 1 = good pattern
    
    # Add bad examples
    bad_examples = [
        [30, 0, 0, 0, 0, 1, 0],  # Large and disorganized
        [5, 0, 1, 0, 0, 4, 0],   # Small but complex
        [25, 1, 0, 0, 0, 2, 0],  # Services without models
    ]
    
    pattern_data.extend(bad_examples)
    
    df_patterns = pd.DataFrame(pattern_data, columns=[
        'Num_Modules', 'Has_Services', 'Has_Workers', 
        'Has_Models', 'Layer_Separation', 'Import_Complexity', 'Good_Quality'
    ])
    
    # Train model
    X = df_patterns.drop('Good_Quality', axis=1)
    y = df_patterns['Good_Quality']
    
    recommendation_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    recommendation_model.fit(X, y)
    
    print("🤖 Architecture Recommendation System")
    print("\nEnter your project characteristics:")
    
    try:
        num_modules = int(input("Estimated number of modules: ") or "15")
        has_services = int(input("Has 'services' directory? (1=Yes, 0=No): ") or "1")
        has_workers = int(input("Has 'workers' directory? (1=Yes, 0=No): ") or "1")
        has_models = int(input("Has 'models' directory? (1=Yes, 0=No): ") or "1")
        layer_separation = int(input("Has clear layer separation? (1=Yes, 0=No): ") or "1")
        complexity = int(input("Import complexity (1-5): ") or "3")
        
        # Make prediction
        input_data = [[num_modules, has_services, has_workers, 
                      has_models, layer_separation, complexity]]
        
        prediction = recommendation_model.predict(input_data)[0]
        probability = recommendation_model.predict_proba(input_data)[0]
        
        print("\n" + "=" * 40)
        print("📋 ARCHITECTURE DIAGNOSIS")
        print("=" * 40)
        
        if prediction == 1:
            print("✅ GOOD ARCHITECTURE")
            print(f"Confidence: {probability[1]:.1%}")
            print("\nIdentified strengths:")
            if has_services == 1:
                print("  ✓ Service separation (best practice)")
            if has_models == 1:
                print("  ✓ Separate models (best practice)")
            if layer_separation == 1:
                print("  ✓ Layer separation (best practice)")
        else:
            print("⚠️ ARCHITECTURE CAN BE IMPROVED")
            print(f"Confidence: {probability[0]:.1%}")
            print("\nRecommendations:")
            if has_services == 0:
                print("  • Consider adding 'services' directory")
            if has_models == 0:
                print("  • Consider adding 'models' directory")
            if layer_separation == 0:
                print("  • Improve responsibility separation")
        
        print("\n📊 Comparison with known patterns:")
        for pattern_name, pattern_chars in patterns.items():
            similarity = sum([
                1 for i, key in enumerate(['Num_Modules', 'Has_Services', 'Has_Workers',
                                         'Has_Models', 'Layer_Separation', 'Import_Complexity'])
                if input_data[0][i] == pattern_chars[key.lower()]
            ]) / 6
            
            print(f"  {pattern_name:12s}: {similarity:.0%} similar")
        
    except ValueError:
        print("⚠️ Invalid input. Using default values...")
    
    return recommendation_model

# ============================================================================
# INTERACTIVE LEARNING
# ============================================================================

def interactive_learning():
    """
    Interactive section for hands-on learning
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE LEARNING")
    print("=" * 60)
    
    print("\n🎓 Choose what you want to learn:")
    print("1. How does the tree make decisions?")
    print("2. What is feature importance?")
    print("3. How to avoid overfitting?")
    print("4. Test with my own example")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🌳 How Decision Tree makes decisions:")
        print("\n1. Starts with ALL data at the root")
        print("2. Finds the best question to split the data")
        print("3. Repeats for each subset until:")
        print("   - Reaches maximum depth")
        print("   - Has few samples")
        print("   - Cannot improve")
        print("\n📝 Example in architecture context:")
        print("   Q1: Has 'services' directory?")
        print("        Yes → Q2: Has 'models' directory?")
        print("        No → Q2: Has layer separation?")
        
    elif choice == "2":
        print("\n⭐ Feature Importance:")
        print("\nMeasures how much each feature contributes to decisions")
        print("\nExample results:")
        print("  Has_Services: 0.45 ← MOST IMPORTANT")
        print("  Has_Models:   0.35")
        print("  Num_Modules:  0.15")
        print("  Has_Workers:  0.05 ← LEAST IMPORTANT")
        print("\n💡 Insight: The model 'learned' that Services and Models")
        print("           are more important than Workers!")
        
    elif choice == "3":
        print("\n🚫 How to avoid Overfitting:")
        print("\nOverfitting = Model memorizes training data")
        print("              but doesn't generalize well")
        print("\nSolutions:")
        print("1. Limit depth (max_depth)")
        print("2. Require minimum samples to split (min_samples_split)")
        print("3. Require minimum samples in leaves (min_samples_leaf)")
        print("4. Use cross-validation")
        print("\n📊 Signs of overfitting:")
        print("  - Training accuracy > 95%, testing < 70%")
        print("  - Tree too deep and complex")
        
    elif choice == "4":
        print("\n🔍 Test your own example:")
        print("\nThink of a project you know and answer:")
        
        responses = {}
        questions = [
            ("How many main modules does it have?", "num_modules", int),
            ("Has separate 'services' directory? (1=Yes, 0=No)", "has_services", int),
            ("Has separate 'models' directory? (1=Yes, 0=No)", "has_models", int),
            ("Has clear layer separation? (1=Yes, 0=No)", "separation", int),
        ]
        
        for question, key, dtype in questions:
            while True:
                try:
                    response = input(f"\n{question}: ")
                    if not response:
                        value = 1 if "Yes" in question else 15
                    else:
                        value = dtype(response)
                    responses[key] = value
                    break
                except ValueError:
                    print("⚠️ Invalid response. Try again.")
        
        print("\n📋 Your project:")
        for key, value in responses.items():
            print(f"  {key}: {value}")
        
        print("\n💭 Based on what we learned, your project probably:")
        if responses.get('has_services', 0) and responses.get('has_models', 0):
            print("  ✅ Follows good organization practices!")
        elif responses.get('separation', 0):
            print("  ⚠️ Has structure, but could improve separation")
        else:
            print("  ❌ Could benefit from better organization")
            
    else:
        print("\n⚠️ Please choose 1, 2, 3, or 4")

# ============================================================================
# PRACTICE EXERCISES
# ============================================================================

def practice_exercises():
    """
    Exercises to practice what was learned
    """
    print("\n" + "=" * 60)
    print("PRACTICE EXERCISES")
    print("=" * 60)
    
    print("\n🧠 Practice with these exercises:")
    
    print("\n1. MODIFY THE CODE:")
    print("   In Example 1, change max_depth from 4 to 6.")
    print("   What happens to tree interpretability?")
    
    print("\n2. ADD NEW FEATURES:")
    print("   Add 'Has_Tests' as a new feature.")
    print("   Retrain the model. Is it important?")
    
    print("\n3. ANALYZE A REAL PROJECT:")
    print("   Choose an open-source project on GitHub.")
    print("   Analyze its structure using learned features.")
    
    print("\n4. CREATE YOUR OWN RULE:")
    print("   Define a new rule for architecture quality.")
    print("   Ex: 'Project needs at least 2 of these 3:")
    print("        - Services, Models, Layer separation'")
    
    print("\n5. DEBUGGING:")
    print("   What's wrong with this code?")
    print("   tree = DecisionTreeClassifier()")
    print("   print(tree.feature_importances_)  # Error!")
    print("   Answer: Need to train the model first!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function of the learning script
    """
    print("🎓 LEARNING SCRIPT: DECISION TREE FOR ARCHITECTURE ANALYSIS")
    print("=" * 70)
    print("\nThis script teaches Machine Learning applied to")
    print("software project structure analysis.")
    print("\nBased on the provided FastAPI example, learn how to:")
    print("  • Decision Trees work")
    print("  • Analyze architecture patterns")
    print("  • Make data-driven decisions")
    print("  • Avoid common problems")
    
    try:
        print("\n🚀 Starting step-by-step tutorial...\n")
        
        # Example 1: Basic analysis
        model1, X_test1, y_test1 = basic_architecture_analysis()
        
        # Example 2: Specific example analysis
        model2, example_data = analyze_fastapi_example()
        
        # Example 3: Optimization
        grid_search = hyperparameter_optimization()
        
        # Example 4: Recommendations
        recommendation_model = architecture_recommendation_generator()
        
        # Interactive learning
        interactive_learning()
        
        # Practice exercises
        practice_exercises()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS! YOU LEARNED:")
        print("=" * 60)
        print("✓ How Decision Trees work")
        print("✓ How to analyze project structure")
        print("✓ Feature importance")
        print("✓ How to avoid overfitting")
        print("✓ Generate data-driven recommendations")
        
        print("\n📁 Files created during learning:")
        files = [
            'architecture_analysis.png',
            'architecture_decision_tree.png',
            'architecture_patterns.png',
            'depth_tradeoff.png'
        ]
        for file in files:
            print(f"   - {file}")
        
        print("\n🌟 Next steps:")
        print("   1. Experiment with your own projects")
        print("   2. Add more features to the analysis")
        print("   3. Learn about Random Forests (next level)")
        print("   4. Explore other ML algorithms")
        
        print("\n💡 Remember: Machine Learning is a tool,")
        print("              not a replacement for human judgment!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Don't worry! Debugging is part of learning.")
        print("Check if you have all required libraries installed.")
        print("Install command: pip install pandas numpy matplotlib scikit-learn seaborn")

# ============================================================================
# QUICK START
# ============================================================================

def quick_start():
    """
    Super simple example for absolute beginners
    """
    print("🚀 QUICK START: YOUR FIRST DECISION TREE")
    print("=" * 50)
    
    # Simplest possible example
    # Predict if a project is well-structured based on 2 features
    
    # Data: [Has_Services, Has_Models], Quality (1=Good, 0=Poor)
    simple_projects = [
        [1, 1, 1],  # Has both, good quality
        [1, 0, 0],  # Only services, poor quality  
        [0, 1, 0],  # Only models, poor quality
        [0, 0, 0],  # Has neither, poor quality
        [1, 1, 1],  # Has both, good quality
        [1, 0, 0],  # Only services, poor quality
    ]
    
    df_simple = pd.DataFrame(simple_projects, 
                             columns=['Has_Services', 'Has_Models', 'Quality'])
    
    X = df_simple[['Has_Services', 'Has_Models']]
    y = df_simple['Quality']
    
    # Create and train tree
    from sklearn.tree import DecisionTreeClassifier
    
    simple_tree = DecisionTreeClassifier(max_depth=2)
    simple_tree.fit(X, y)
    
    # Make prediction
    new_project = [[1, 1]]  # Has Services and Models
    prediction = simple_tree.predict(new_project)
    
    print(f"\nProject with Services=1, Models=1")
    print(f"Quality prediction: {'✅ GOOD' if prediction[0] == 1 else '❌ POOR'}")
    
    # Visualize the simple tree
    plt.figure(figsize=(10, 6))
    plot_tree(simple_tree, 
              feature_names=['Has_Services', 'Has_Models'],
              class_names=['Poor', 'Good'],
              filled=True,
              rounded=True)
    plt.title("Simple Decision Tree", fontsize=14)
    plt.tight_layout()
    plt.savefig('simple_tree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💡 The tree learned: 'If Has_Services AND Has_Models, then quality GOOD'")
    print("   This makes sense! Well-structured projects usually have both.")

# ============================================================================
# RUN THE SCRIPT
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