import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 80)
print("TEACHING SOLID PRINCIPLES IN MACHINE LEARNING")
print("=" * 80)

# ============================================================================
# PRINCIPLE 1: SINGLE RESPONSIBILITY PRINCIPLE (SRP)
# ============================================================================
print("\n" + "=" * 80)
print("1. SINGLE RESPONSIBILITY PRINCIPLE (SRP)")
print("=" * 80)
print("Each class should have only ONE reason to change")
print("-" * 80)


class MLModelSRPViolation:
        
    def __init__(self):
        self.model = None
        self.data = None
    
    def load_data(self, filepath):  
        self.data = pd.read_csv(filepath)
    
    def preprocess(self):  # Responsibility 2: Preprocess
        self.data = self.data.fillna(self.data.mean())
    
    def train(self):  # Responsibility 3: Train model
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)
    
    def evaluate(self):  # Responsibility 4: Evaluate
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)
    
    def save_model(self, path):  # Responsibility 5: Save model
        import joblib
        joblib.dump(self.model, path)

print("\n❌ SRP VIOLATION (Before):")
print("- One class doing 5 different things")
print("- If data format changes, I need to modify this class")
print("- If evaluation method changes, I need to modify this class")
print("- If persistence method changes, I need to modify this class")
print("\nProblem: TOO MANY REASONS TO CHANGE!")


class DataLoader:
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)
    
    def load_from_database(self, query: str) -> pd.DataFrame:
        
        pass

class DataPreprocessor:
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mean())
    
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        
        return (data - data.mean()) / data.std()

class ModelTrainer:
        
    def train_decision_tree(self, X, y, **params):
        model = DecisionTreeClassifier(**params)
        model.fit(X, y)
        return model
    
    def train_random_forest(self, X, y, **params):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        return model

class ModelEvaluator:
        
    def calculate_accuracy(self, model, X, y) -> float:
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)
    
    def calculate_confusion_matrix(self, model, X, y):
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X)
        return confusion_matrix(y, y_pred)

class ModelSaver:
        
    def save_to_disk(self, model, path: str):
        import joblib
        joblib.dump(model, path)
    
    def save_to_cloud(self, model, cloud_path: str):
        # Cloud implementation
        pass

print("\n CORRECT SRP APPLICATION (After):")
print("- Each class has ONE well-defined responsibility")
print("- If data format changes, I only modify DataLoader")
print("- If evaluation method changes, I only modify ModelEvaluator")
print("- If persistence method changes, I only modify ModelSaver")
print("\nBenefit: ONE REASON TO CHANGE PER CLASS!")

# ============================================================================
# PRINCIPLE 2: OPEN/CLOSED PRINCIPLE (OCP)
# ============================================================================
print("\n\n" + "=" * 80)
print("2. OPEN/CLOSED PRINCIPLE (OCP)")
print("=" * 80)
print("Entities should be OPEN for extension, but CLOSED for modification")
print("-" * 80)


class ModelProcessorOCPViolation:
        
    def process_model(self, model_type: str, X, y):
        if model_type == "decision_tree":
            model = DecisionTreeClassifier()
            model.fit(X, y)
            return model
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            model.fit(X, y)
            return model
        
        # elif model_type == "svm":
        #     from sklearn.svm import SVC
        #     model = SVC()
        #     model.fit(X, y)
        #     return model
        else:
            raise ValueError(f"Model {model_type} not supported")

print("\n OCP VIOLATION (Before):")
print("- To add new model type (SVM, XGBoost, etc)")
print("- I need to MODIFY existing class")
print("- Risk of breaking existing functionality")


class BaseModel(ABC):
        
    @abstractmethod
    def train(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

class DecisionTreeModel(BaseModel):
        
    def __init__(self, **params):
        self.model = DecisionTreeClassifier(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel(BaseModel):
        
    def __init__(self, **params):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class SVMModel(BaseModel):
        
    def __init__(self, **params):
        from sklearn.svm import SVC
        self.model = SVC(**params)
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class ModelProcessorOCP:
        
    def process_model(self, model: BaseModel, X, y):
        
        trained_model = model.train(X, y)
        return trained_model

print("\n CORRECT OCP APPLICATION (After):")
print("- To add new model, just create new class")
print("- DON'T need to modify existing code")
print("- Base class is CLOSED for modification")
print("- System is OPEN for extension")


print("\n PRACTICAL DEMONSTRATION:")
print("1. Creating DecisionTree (existing):")
dt_model = DecisionTreeModel(max_depth=5)
print("   ✓ DecisionTree created without modifying code")

print("\n2. Adding SVM (new):")
svm_model = SVMModel(kernel='rbf')
print("   ✓ SVM added JUST by creating new class")

print("\n3. ModelProcessor works with BOTH:")
processor = ModelProcessorOCP()

print("   ✓ Processor accepts any model inheriting from BaseModel")

# ============================================================================
# PRINCIPLE 3: LISKOV SUBSTITUTION PRINCIPLE (LSP)
# ============================================================================
print("\n\n" + "=" * 80)
print("3. LISKOV SUBSTITUTION PRINCIPLE (LSP)")
print("=" * 80)
print("Subclasses should be substitutable for their base classes")
print("-" * 80)


class BaseClassifier(ABC):
        
    @abstractmethod
    def train(self, X, y):
        pass
    
    def predict_proba(self, X):
        """Returns class probabilities"""
        raise NotImplementedError

class GoodDecisionTree(BaseClassifier):
        
    def __init__(self):
        self.model = DecisionTreeClassifier()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict_proba(self, X):
        
        return self.model.predict_proba(X)

class BadDecisionTree(BaseClassifier):
        
    def __init__(self):
        self.model = DecisionTreeClassifier()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict_proba(self, X):
        
        raise NotImplementedError("This model doesn't support probabilities")
    
    def predict(self, X):
        
        return self.model.predict(X)

print("\n LSP VIOLATION (Example):")
print("- Client expects predict_proba() to work for ANY BaseClassifier")
print("- BadDecisionTree breaks this expectation")
print("- Cannot be substituted for base class without breaking system")

print("\n CORRECT LSP APPLICATION:")
print("- Any subclass of BaseClassifier should:")
print("  1. Implement ALL abstract methods")
print("  2. NOT throw unexpected exceptions")
print("  3. Follow expected contract/behavior")


def evaluate_classifier(classifier: BaseClassifier, X_test, y_test):
    
    try:
        
        probas = classifier.predict_proba(X_test)
        print(f"  ✓ Probabilities calculated: {probas.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

print("\n TESTING SUBSTITUTION:")
print("Testing GoodDecisionTree (should work):")
good_dt = GoodDecisionTree()


print("\nTesting BadDecisionTree (breaks system):")
bad_dt = BadDecisionTree()


# ============================================================================
# PRINCIPLE 4: INTERFACE SEGREGATION PRINCIPLE (ISP)
# ============================================================================
print("\n\n" + "=" * 80)
print("4. INTERFACE SEGREGATION PRINCIPLE (ISP)")
print("=" * 80)
print("Many specific interfaces are better than one general interface")
print("-" * 80)


class MLProcessorISPViolation(ABC):
        
    @abstractmethod
    def load_data(self): pass
    
    @abstractmethod
    def preprocess(self): pass
    
    @abstractmethod
    def train_model(self): pass
    
    @abstractmethod
    def evaluate_model(self): pass
    
    @abstractmethod
    def deploy_model(self): pass  # ❌ Many clients don't need this!
    
    @abstractmethod
    def monitor_model(self): pass  # ❌ Most don't need this!
    
    @abstractmethod
    def retrain_model(self): pass  # ❌ Specific to some cases!

class SimpleTrainerViolation(MLProcessorISPViolation):
        
    def load_data(self):
        return pd.DataFrame()
    
    def preprocess(self):
        pass
    
    def train_model(self):
        pass
    
    def evaluate_model(self):
        pass
    
    def deploy_model(self):
        
        raise NotImplementedError("Don't handle deployment")
    
    def monitor_model(self):
        
        raise NotImplementedError("Don't do monitoring")
    
    def retrain_model(self):
        
        raise NotImplementedError("Don't do retraining")

print("\n ISP VIOLATION (Before):")
print("- Single interface with 7 methods")
print("- SimpleTrainer only uses 4 methods, but forced to implement 7")
print("- High coupling, low cohesion")


class DataLoaderInterface(ABC):
    
    
    @abstractmethod
    def load_data(self, source) -> pd.DataFrame:
        pass

class DataPreprocessorInterface(ABC):
        
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class ModelTrainerInterface(ABC):
        
    @abstractmethod
    def train(self, X, y, **params):
        pass

class ModelEvaluatorInterface(ABC):
        
    @abstractmethod
    def evaluate(self, model, X, y) -> Dict[str, float]:
        pass


class ModelDeployerInterface(ABC):
        
    @abstractmethod
    def deploy(self, model, endpoint: str):
        pass

class ModelMonitorInterface(ABC):
        
    @abstractmethod
    def monitor(self, model, metrics: List[str]):
        pass


class SimpleTrainerISP(ModelTrainerInterface, ModelEvaluatorInterface):
        
    def train(self, X, y, **params):
        model = DecisionTreeClassifier(**params)
        model.fit(X, y)
        return model
    
    def evaluate(self, model, X, y) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        y_pred = model.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted')
        }

class FullMLPipeline(DataLoaderInterface, DataPreprocessorInterface, 
                     ModelTrainerInterface, ModelEvaluatorInterface,
                     ModelDeployerInterface):
    
    
    def load_data(self, source) -> pd.DataFrame:
        return pd.read_csv(source)
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mean())
    
    def train(self, X, y, **params):
        model = DecisionTreeClassifier(**params)
        model.fit(X, y)
        return model
    
    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        return {'accuracy': accuracy_score(y, y_pred)}
    
    def deploy(self, model, endpoint: str):
        print(f"Model deployed at {endpoint}")

print("\n CORRECT ISP APPLICATION (After):")
print("- 6 specific interfaces instead of 1 general interface")
print("- SimpleTrainer implements ONLY 2 interfaces")
print("- FullMLPipeline implements 5 interfaces")
print("- Each class implements ONLY what it needs")

# ============================================================================
# PRINCIPLE 5: DEPENDENCY INVERSION PRINCIPLE (DIP)
# ============================================================================
print("\n\n" + "=" * 80)
print("5. DEPENDENCY INVERSION PRINCIPLE (DIP)")
print("=" * 80)
print("Depend on ABSTRACTIONS, not on concrete implementations")
print("-" * 80)


class MLPipelineDIPViolation:
    """DIP Violation - Strong coupling to concrete implementations"""
    
    def __init__(self):
        
        self.data_loader = CSVDataLoader()  # Concrete
        self.preprocessor = StandardScalerPreprocessor()  # Concrete
        self.trainer = SklearnDecisionTreeTrainer()  # Concrete
    
    def run(self, csv_file: str):
        
        data = self.data_loader.load_csv(csv_file)
        data = self.preprocessor.scale(data)
        model = self.trainer.train(data)
        return model

class CSVDataLoader:
    """Concrete implementation - loads only CSV"""
    def load_csv(self, path: str):
        return pd.read_csv(path)

class StandardScalerPreprocessor:
    
    def scale(self, data: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(data)

class SklearnDecisionTreeTrainer:
    
    def train(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model

print("\n DIP VIOLATION (Before):")
print("- MLPipeline knows concrete IMPLEMENTATIONS")
print("- Hard to switch from CSV to database")
print("- Hard to switch from StandardScaler to MinMaxScaler")
print("- Hard to switch from DecisionTree to RandomForest")
print("- Hard to test (concrete dependencies)")


class IDataLoader(ABC):
        
    @abstractmethod
    def load(self, source) -> pd.DataFrame:
        pass

class IDataPreprocessor(ABC):
        
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class IModelTrainer(ABC):
        
    @abstractmethod
    def train(self, X, y, **params):
        pass


class CSVDataLoaderDIP(IDataLoader):
        
    def load(self, source) -> pd.DataFrame:
        return pd.read_csv(source)

class DatabaseDataLoader(IDataLoader):
        
    def load(self, source) -> pd.DataFrame:
        import sqlite3
        conn = sqlite3.connect(source)
        return pd.read_sql_query("SELECT * FROM data", conn)

class StandardScalerPreprocessorDIP(IDataPreprocessor):
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import StandardScaler
        return pd.DataFrame(
            StandardScaler().fit_transform(data),
            columns=data.columns
        )

class MinMaxPreprocessor(IDataPreprocessor):
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import MinMaxScaler
        return pd.DataFrame(
            MinMaxScaler().fit_transform(data),
            columns=data.columns
        )

class DecisionTreeTrainerDIP(IModelTrainer):
        
    def train(self, X, y, **params):
        model = DecisionTreeClassifier(**params)
        model.fit(X, y)
        return model

class RandomForestTrainer(IModelTrainer):
        
    def train(self, X, y, **params):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        return model


class MLPipelineDIP:
        
    def __init__(self, 
                 data_loader: IDataLoader,
                 preprocessor: IDataPreprocessor,
                 trainer: IModelTrainer):
        
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.trainer = trainer
    
    def run(self, data_source: str, target_column: str = 'target'):
        
        data = self.data_loader.load(data_source)
        
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        
        X_processed = self.preprocessor.process(X)
        
        
        model = self.trainer.train(X_processed, y)
        
        return model

print("\n CORRECT DIP APPLICATION (After):")
print("- MLPipeline knows only INTERFACES/ABSTRACTIONS")
print("- Easy to swap implementations via dependency injection")
print("- Easy to test (using mocks/stubs)")
print("- Low coupling, high flexibility")


print("\n FLEXIBILITY DEMONSTRATION:")
print("1. Pipeline with CSV + StandardScaler + DecisionTree:")
pipeline1 = MLPipelineDIP(
    data_loader=CSVDataLoaderDIP(),
    preprocessor=StandardScalerPreprocessorDIP(),
    trainer=DecisionTreeTrainerDIP()
)
print("   ✓ Configured")

print("\n2. Pipeline with Database + MinMax + RandomForest:")
pipeline2 = MLPipelineDIP(
    data_loader=DatabaseDataLoader(),
    preprocessor=MinMaxPreprocessor(),
    trainer=RandomForestTrainer()
)
print("   ✓ Configured without modifying MLPipeline")

print("\n3. Pipeline for TESTING (using mocks):")
class MockDataLoader(IDataLoader):
    def load(self, source):
        return pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })

class MockPreprocessor(IDataPreprocessor):
    def process(self, data):
        return data  

class MockTrainer(IModelTrainer):
    def train(self, X, y, **params):
        from unittest.mock import Mock
        mock_model = Mock()
        mock_model.predict.return_value = [0, 1, 0]
        return mock_model

test_pipeline = MLPipelineDIP(
    data_loader=MockDataLoader(),
    preprocessor=MockPreprocessor(),
    trainer=MockTrainer()
)
print("   ✓ Test pipeline created easily")

# ============================================================================
# COMPLETE EXAMPLE: DECISION TREE WITH ALL SOLID PRINCIPLES
# ============================================================================
print("\n\n" + "=" * 80)
print("COMPLETE EXAMPLE: DECISION TREE WITH ALL SOLID PRINCIPLES")
print("=" * 80)


class IFeatureEngineer(ABC):
    """Single Responsibility: Feature engineering"""
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class IFeatureSelector(ABC):
    
    @abstractmethod
    def select_features(self, data: pd.DataFrame, target: str) -> List[str]:
        pass

class IHyperparameterTuner(ABC):
    
    @abstractmethod
    def tune(self, model, X, y, param_grid: Dict) -> Dict:
        pass


class PolynomialFeatureEngineer(IFeatureEngineer):
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(data.select_dtypes(include=[np.number]))
        poly_df = pd.DataFrame(
            poly_features,
            columns=poly.get_feature_names_out(data.select_dtypes(include=[np.number]).columns)
        )
        return pd.concat([data, poly_df], axis=1)

class RFEFeatureSelector(IFeatureSelector):
        
    def select_features(self, data: pd.DataFrame, target: str) -> List[str]:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        
        X = data.drop(target, axis=1)
        y = data[target]
        
        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=10)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        return selected_features

class GridSearchTuner(IHyperparameterTuner):
        
    def tune(self, model, X, y, param_grid: Dict) -> Dict:
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }


class SolidDecisionTreePipeline:
        
    def __init__(self,
                 feature_engineer: IFeatureEngineer,
                 feature_selector: IFeatureSelector,
                 data_preprocessor: IDataPreprocessor,
                 model_trainer: IModelTrainer,
                 hyperparameter_tuner: Optional[IHyperparameterTuner] = None):
        
        
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.data_preprocessor = data_preprocessor
        self.model_trainer = model_trainer
        self.hyperparameter_tuner = hyperparameter_tuner
        
        
        self.data = None
        self.selected_features = None
        self.model = None
        self.tuning_results = None
    
    def load_and_prepare(self, data_source: str, target_column: str):
        
        
        self.data = pd.read_csv(data_source)
        
        
        self.data = self.feature_engineer.engineer_features(self.data)
        
        
        self.selected_features = self.feature_selector.select_features(
            self.data, target_column
        )
                
        self.selected_features.append(target_column)
        self.data = self.data[self.selected_features]
        
        return self.data
    
    def train_model(self, target_column: str, tune_hyperparams: bool = False):
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_prepare() first.")
        
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        
        X_processed = self.data_preprocessor.process(X)
        
        if tune_hyperparams and self.hyperparameter_tuner:
            
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = DecisionTreeClassifier(random_state=42)
            self.tuning_results = self.hyperparameter_tuner.tune(
                base_model, X_processed, y, param_grid
            )
            self.model = self.tuning_results['best_model']
        else:
            
            self.model = self.model_trainer.train(X_processed, y)
        
        return self.model
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str):
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        X_test = test_data[self.selected_features].drop(target_column, axis=1)
        y_test = test_data[target_column]
        
        X_test_processed = self.data_preprocessor.process(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = self.model.predict(X_test_processed)
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'accuracy': accuracy_score(y_test, y_pred)
        }


print("\n CREATING SOLID PIPELINE:")
print("1. Instantiating components with single responsibilities (SRP):")
feature_engineer = PolynomialFeatureEngineer()
feature_selector = RFEFeatureSelector()
preprocessor = StandardScalerPreprocessorDIP()
trainer = DecisionTreeTrainerDIP()
tuner = GridSearchTuner()

print("   ✓ Components created")

print("\n2. Creating pipeline (DIP + ISP):")
pipeline = SolidDecisionTreePipeline(
    feature_engineer=feature_engineer,
    feature_selector=feature_selector,
    data_preprocessor=preprocessor,
    model_trainer=trainer,
    hyperparameter_tuner=tuner
)
print("   ✓ Pipeline created with dependency injection")

print("\n3. Pipeline is extensible (OCP):")
print("   - Can create new FeatureEngineer without modifying pipeline")
print("   - Can create new FeatureSelector without modifying pipeline")
print("   - Can create new HyperparameterTuner without modifying pipeline")

print("\n4. Components are substitutable (LSP):")
print("   - Any IFeatureEngineer works")
print("   - Any IFeatureSelector works")
print("   - Any IModelTrainer works")

print("\n5. Each class has specific interface (ISP):")
print("   - IFeatureEngineer: only engineer_features()")
print("   - IFeatureSelector: only select_features()")
print("   - IHyperparameterTuner: only tune()")

# ============================================================================
# SUMMARY AND BENEFITS
# ============================================================================
print("\n\n" + "=" * 80)
print("SOLID PRINCIPLES BENEFITS SUMMARY")
print("=" * 80)

print("\n SINGLE RESPONSIBILITY:")
print("   - Code easier to understand")
print("   - Fewer bugs when modifying")
print("   - Better organization")
print("   - Facilitates unit testing")

print("\n OPEN/CLOSED:")
print("   - Add functionality without modifying existing code")
print("   - Lower regression risk")
print("   - More stable system")
print("   - Facilitates extensibility")

print("\n LISKOV SUBSTITUTION:")
print("   - Safe object substitution")
print("   - Reliable polymorphism")
print("   - Well-defined contracts")
print("   - Fewer unexpected exceptions")

print("\n INTERFACE SEGREGATION:")
print("   - Specific, focused interfaces")
print("   - Clients not forced to implement unused methods")
print("   - Better contract organization")
print("   - Lower coupling")

print("\n DEPENDENCY INVERSION:")
print("   - Low coupling between modules")
print("   - Easy to test (mocks/stubs)")
print("   - Easy to modify (dependency injection)")
print("   - Flexibility to swap implementations")

print("\n" + "=" * 80)
print("SUCCESS! Now you understand SOLID with Decision Trees! ")
print("=" * 80)