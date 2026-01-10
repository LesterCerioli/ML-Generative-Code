import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=" * 80)
print("TEACHING SOLID PRINCIPLES IN MACHINE LEARNING - LINEAR REGRESSION")
print("=" * 80)

# ============================================================================
# PRINCIPLE 1: SINGLE RESPONSIBILITY PRINCIPLE (SRP)
# ============================================================================
print("\n" + "=" * 80)
print("1. SINGLE RESPONSIBILITY PRINCIPLE (SRP)")
print("=" * 80)
print("Each class should have only ONE reason to change")
print("-" * 80)


class RegressionModelSRPViolation:
        
    def __init__(self):
        self.model = None
        self.data = None
        self.scaler = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
    
    def preprocess(self):
        self.data = self.data.dropna()
    
    def scale_features(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
    
    def train(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        self.model = LinearRegression()
        self.model.fit(X, y)
    
    def evaluate(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        y_pred = self.model.predict(X)
        return mean_squared_error(y, y_pred)
    
    def plot_results(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        y_pred = self.model.predict(X)
        plt.scatter(y, y_pred)
        plt.show()

print("\n SRP VIOLATION (Before):")
print("- One class doing 6 different things")
print("- If data loading changes, modify this class")
print("- If evaluation metric changes, modify this class")
print("- If plotting changes, modify this class")
print("\nProblem: TOO MANY REASONS TO CHANGE!")


class DataLoader:
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)
    
    def load_from_api(self, url: str) -> pd.DataFrame:
        
        pass

class DataCleaner:
        
    def remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()
    
    def remove_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]
        return data

class FeatureScaler:
        
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.scaler.transform(data),
            columns=data.columns,
            index=data.index
        )

class RegressionTrainer:
        
    def train_linear_regression(self, X, y, **params):
        model = LinearRegression(**params)
        model.fit(X, y)
        return model
    
    def train_ridge_regression(self, X, y, alpha=1.0):
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        return model
    
    def train_lasso_regression(self, X, y, alpha=1.0):
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        return model

class ModelEvaluator:
        
    def calculate_mse(self, model, X, y) -> float:
        y_pred = model.predict(X)
        return mean_squared_error(y, y_pred)
    
    def calculate_r2(self, model, X, y) -> float:
        y_pred = model.predict(X)
        return r2_score(y, y_pred)
    
    def calculate_mae(self, model, X, y) -> float:
        from sklearn.metrics import mean_absolute_error
        y_pred = model.predict(X)
        return mean_absolute_error(y, y_pred)

class ResultVisualizer:
        
    def plot_predictions_vs_actual(self, y_true, y_pred, title="Predictions vs Actual"):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title="Residual Plot"):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.grid(True)
        plt.show()

print("\n CORRECT SRP APPLICATION (After):")
print("- Each class has ONE well-defined responsibility")
print("- If data loading changes, only modify DataLoader")
print("- If evaluation changes, only modify ModelEvaluator")
print("- If visualization changes, only modify ResultVisualizer")
print("\nBenefit: ONE REASON TO CHANGE PER CLASS!")

# ============================================================================
# PRINCIPLE 2: OPEN/CLOSED PRINCIPLE (OCP)
# ============================================================================
print("\n\n" + "=" * 80)
print("2. OPEN/CLOSED PRINCIPLE (OCP)")
print("=" * 80)
print("Entities should be OPEN for extension, but CLOSED for modification")
print("-" * 80)


class RegressionProcessorOCPViolation:
    """OCP Violation - Needs modification for new regression types"""
    
    def create_regression_model(self, model_type: str, **params):
        if model_type == "linear":
            return LinearRegression(**params)
        elif model_type == "ridge":
            return Ridge(**params)
        elif model_type == "lasso":
            return Lasso(**params)
        
        # elif model_type == "elastic_net":
        #     from sklearn.linear_model import ElasticNet
        #     return ElasticNet(**params)
        else:
            raise ValueError(f"Model type {model_type} not supported")

print("\n OCP VIOLATION (Before):")
print("- To add ElasticNet, need to modify existing class")
print("- Risk of breaking existing functionality")
print("- Violates 'closed for modification'")


class BaseRegressionModel(ABC):
        
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def get_coefficients(self):
        pass

class LinearRegressionModel(BaseRegressionModel):
        
    def __init__(self, **params):
        self.model = LinearRegression(**params)
        self.coefficients_ = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coefficients_ = self.model.coef_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_coefficients(self):
        return self.coefficients_

class RidgeRegressionModel(BaseRegressionModel):
        
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.coefficients_ = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coefficients_ = self.model.coef_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_coefficients(self):
        return self.coefficients_


class ElasticNetRegressionModel(BaseRegressionModel):
        
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.coefficients_ = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coefficients_ = self.model.coef_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_coefficients(self):
        return self.coefficients_

class RegressionFactory:
        
    @staticmethod
    def create_model(model_type: str, **params) -> BaseRegressionModel:
        
        registry = {
            "linear": LinearRegressionModel,
            "ridge": RidgeRegressionModel,
            "lasso": LinearRegressionModel,  # Using LinearRegression for demo
        }
        
        model_class = registry.get(model_type)
        if model_class:
            return model_class(**params)
        raise ValueError(f"Model type {model_type} not found in registry")

print("\n CORRECT OCP APPLICATION (After):")
print("- To add new model, just register it in factory")
print("- DON'T need to modify existing factory method")
print("- Base class is CLOSED for modification")
print("- System is OPEN for extension via registry")

# ============================================================================
# PRINCIPLE 3: LISKOV SUBSTITUTION PRINCIPLE (LSP)
# ============================================================================
print("\n\n" + "=" * 80)
print("3. LISKOV SUBSTITUTION PRINCIPLE (LSP)")
print("=" * 80)
print("Subclasses should be substitutable for their base classes")
print("-" * 80)


class BaseRegressor(ABC):
        
    @abstractmethod
    def train(self, X, y):
        pass
    
    def get_feature_importance(self):
        """Returns feature importance scores"""
        raise NotImplementedError("Feature importance not implemented")
    
    def get_prediction_intervals(self, X, confidence=0.95):
        
        raise NotImplementedError("Prediction intervals not implemented")

class GoodLinearRegressor(BaseRegressor):
        
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def get_feature_importance(self):
        
        return np.abs(self.model.coef_)
    
    def get_prediction_intervals(self, X, confidence=0.95):
        
        y_pred = self.model.predict(X)
        
        residuals = self.model._residues if hasattr(self.model, '_residues') else 0.1
        interval = residuals * (1 + confidence)
        return y_pred - interval, y_pred + interval

class BadLinearRegressor(BaseRegressor):
        
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self
    
    def get_feature_importance(self):
        
        return "Feature importance not available for this model"
    
    def get_prediction_intervals(self, X, confidence=0.95):
        
        raise RuntimeError("This model doesn't support confidence intervals")

print("\n LSP VIOLATION (Example):")
print("- Client expects get_feature_importance() to return array")
print("- BadLinearRegressor returns string instead")
print("- Client code breaks when substituting BadLinearRegressor")

print("\n CORRECT LSP APPLICATION:")
print("- All subclasses must:")
print("  1. Return same types as base class methods")
print("  2. Not throw unexpected exceptions")
print("  3. Maintain same pre/post conditions")

# ============================================================================
# PRINCIPLE 4: INTERFACE SEGREGATION PRINCIPLE (ISP)
# ============================================================================
print("\n\n" + "=" * 80)
print("4. INTERFACE SEGREGATION PRINCIPLE (ISP)")
print("=" * 80)
print("Many specific interfaces are better than one general interface")
print("-" * 80)


class RegressionPipelineISPViolation(ABC):
        
    @abstractmethod
    def load_and_clean_data(self): pass
    
    @abstractmethod
    def engineer_features(self): pass
    
    @abstractmethod
    def train_model(self): pass
    
    @abstractmethod
    def evaluate_model(self): pass
    
    @abstractmethod
    def hyperparameter_tune(self): pass  
    
    @abstractmethod
    def cross_validate(self): pass  
    
    @abstractmethod
    def deploy_to_production(self): pass  

class SimpleLinearRegressionViolation(RegressionPipelineISPViolation):
        
    def load_and_clean_data(self):
        return pd.DataFrame()
    
    def engineer_features(self):
        pass
    
    def train_model(self):
        pass
    
    def evaluate_model(self):
        pass
    
    def hyperparameter_tune(self):
        
        raise NotImplementedError("No hyperparameter tuning needed")
    
    def cross_validate(self):
        
        raise NotImplementedError("No cross-validation needed")
    
    def deploy_to_production(self):
        
        raise NotImplementedError("No deployment needed")

print("\n ISP VIOLATION (Before):")
print("- Single interface with 7 methods")
print("- SimpleLinearRegression only needs 4 methods")
print("- Forced to implement 3 unused methods")


class IDataPipeline(ABC):
        
    @abstractmethod
    def process_data(self, source) -> pd.DataFrame:
        pass

class IFeatureEngineering(ABC):
        
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class IModelTraining(ABC):
        
    @abstractmethod
    def train(self, X, y, **params):
        pass

class IModelEvaluation(ABC):
        
    @abstractmethod
    def evaluate(self, model, X, y) -> Dict[str, float]:
        pass

# Optional interfaces
class IHyperparameterTuning(ABC):
        
    @abstractmethod
    def tune(self, model, X, y, param_grid: Dict) -> Dict:
        pass

class ICrossValidation(ABC):
        
    @abstractmethod
    def cross_validate(self, model, X, y, cv: int = 5) -> Dict:
        pass

class IModelDeployment(ABC):
        
    @abstractmethod
    def deploy(self, model, endpoint: str):
        pass


class SimpleLinearRegressionPipeline(IModelTraining, IModelEvaluation):
        
    def train(self, X, y, **params):
        model = LinearRegression(**params)
        model.fit(X, y)
        return model
    
    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mae': np.mean(np.abs(y - y_pred))
        }

class FullRegressionPipeline(IDataPipeline, IFeatureEngineering, 
                            IModelTraining, IModelEvaluation,
                            IHyperparameterTuning, ICrossValidation):
    
    
    def process_data(self, source) -> pd.DataFrame:
        data = pd.read_csv(source)
        return data.dropna()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[f'{col}_squared'] = data[col] ** 2
        return data
    
    def train(self, X, y, **params):
        model = LinearRegression(**params)
        model.fit(X, y)
        return model
    
    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        return {'mse': mean_squared_error(y, y_pred)}
    
    def tune(self, model, X, y, param_grid: Dict) -> Dict:
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y)
        return {'best_params': grid_search.best_params_}
    
    def cross_validate(self, model, X, y, cv: int = 5) -> Dict:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return {'mean_score': scores.mean(), 'std_score': scores.std()}

print("\n CORRECT ISP APPLICATION (After):")
print("- 7 specific interfaces instead of 1 general interface")
print("- SimplePipeline implements ONLY 2 interfaces")
print("- FullPipeline implements 6 interfaces")
print("- Each class implements ONLY what it needs")

# ============================================================================
# PRINCIPLE 5: DEPENDENCY INVERSION PRINCIPLE (DIP)
# ============================================================================
print("\n\n" + "=" * 80)
print("5. DEPENDENCY INVERSION PRINCIPLE (DIP)")
print("=" * 80)
print("Depend on ABSTRACTIONS, not on concrete implementations")
print("-" * 80)


class LinearRegressionPipelineDIPViolation:
    """DIP Violation - Strong coupling to concrete implementations"""
    
    def __init__(self):
        
        self.data_loader = CSVDataLoader()  # Concrete
        self.feature_engineer = PolynomialFeatureEngineer()  # Concrete
        self.trainer = SklearnLinearRegressionTrainer()  # Concrete
        self.evaluator = MSEEvaluator()  # Concrete
    
    def run(self, csv_file: str, target_column: str):
        
        data = self.data_loader.load_csv(csv_file)
        data = self.feature_engineer.add_polynomial_features(data)
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        model = self.trainer.train_sklearn(X, y)
        score = self.evaluator.calculate_mse(model, X, y)
        return model, score

class CSVDataLoader:
    def load_csv(self, path: str):
        return pd.read_csv(path)

class PolynomialFeatureEngineer:
    def add_polynomial_features(self, data: pd.DataFrame):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        poly_features = poly.fit_transform(data[numeric_cols])
        return pd.concat([data, pd.DataFrame(poly_features, columns=[f'poly_{i}' for i in range(poly_features.shape[1])])], axis=1)

class SklearnLinearRegressionTrainer:
    def train_sklearn(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

class MSEEvaluator:
    def calculate_mse(self, model, X, y):
        y_pred = model.predict(X)
        return mean_squared_error(y, y_pred)

print("\n DIP VIOLATION (Before):")
print("- Pipeline knows concrete IMPLEMENTATIONS")
print("- Hard to switch from CSV to database")
print("- Hard to switch from Polynomial to other feature engineering")
print("- Hard to switch from MSE to R² evaluation")


class IDataLoader(ABC):
        
    @abstractmethod
    def load(self, source) -> pd.DataFrame:
        pass

class IFeatureEngineer(ABC):
       
    @abstractmethod
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class IRegressionTrainer(ABC):
        
    @abstractmethod
    def train(self, X, y, **params):
        pass

class IRegressionEvaluator(ABC):
        
    @abstractmethod
    def evaluate(self, model, X, y) -> Dict[str, float]:
        pass

# Concrete implementations
class CSVDataLoaderDIP(IDataLoader):
    def load(self, source) -> pd.DataFrame:
        return pd.read_csv(source)

class DatabaseDataLoader(IDataLoader):
    def load(self, source) -> pd.DataFrame:
        
        import sqlite3
        conn = sqlite3.connect(source)
        return pd.read_sql_query("SELECT * FROM data", conn)

class PolynomialFeatureEngineerDIP(IFeatureEngineer):
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        poly_features = poly.fit_transform(data[numeric_cols])
        feature_names = poly.get_feature_names_out(numeric_cols)
        return pd.concat([data, pd.DataFrame(poly_features, columns=feature_names)], axis=1)

class InteractionFeatureEngineer(IFeatureEngineer):
    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        return data

class LinearRegressionTrainerDIP(IRegressionTrainer):
    def train(self, X, y, **params):
        model = LinearRegression(**params)
        model.fit(X, y)
        return model

class RidgeRegressionTrainer(IRegressionTrainer):
    def train(self, X, y, **params):
        model = Ridge(**params)
        model.fit(X, y)
        return model

class ComprehensiveEvaluator(IRegressionEvaluator):
    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'explained_variance': np.var(y_pred) / np.var(y) if np.var(y) > 0 else 0
        }

class SimpleEvaluator(IRegressionEvaluator):
    def evaluate(self, model, X, y) -> Dict[str, float]:
        y_pred = model.predict(X)
        return {'mse': mean_squared_error(y, y_pred)}

# Pipeline that depends on ABSTRACTIONS
class LinearRegressionPipelineDIP:
    """ CORRECT - Depends only on ABSTRACTIONS"""
    
    def __init__(self, 
                 data_loader: IDataLoader,
                 feature_engineer: IFeatureEngineer,
                 trainer: IRegressionTrainer,
                 evaluator: IRegressionEvaluator):
        #  DEPENDS on ABSTRACTIONS
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.trainer = trainer
        self.evaluator = evaluator
    
    def run(self, data_source: str, target_column: str, **train_params):
        
        data = self.data_loader.load(data_source)
                
        data = self.feature_engineer.engineer(data)
                
        X = data.drop(target_column, axis=1)
        y = data[target_column]
                
        model = self.trainer.train(X, y, **train_params)
                
        metrics = self.evaluator.evaluate(model, X, y)
        
        return model, metrics

print("\n CORRECT DIP APPLICATION (After):")
print("- Pipeline knows only INTERFACES")
print("- Easy to swap implementations via dependency injection")
print("- Easy to test (using mocks)")
print("- Low coupling, high flexibility")

# ============================================================================
# COMPLETE EXAMPLE: LINEAR REGRESSION WITH ALL SOLID PRINCIPLES
# ============================================================================
print("\n\n" + "=" * 80)
print("COMPLETE EXAMPLE: LINEAR REGRESSION WITH ALL SOLID PRINCIPLES")
print("=" * 80)


class IDataValidator(ABC):
    """Single Responsibility: Validate data"""
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        pass

class IFeatureSelector(ABC):
    """Single Responsibility: Select features"""
    @abstractmethod
    def select(self, data: pd.DataFrame, target: str, n_features: int) -> List[str]:
        pass

class IModelInterpreter(ABC):
    """Single Responsibility: Interpret model"""
    @abstractmethod
    def interpret(self, model, feature_names: List[str]) -> Dict[str, Any]:
        pass


class DataQualityValidator(IDataValidator):
        
    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = []
                
        missing = data.isnull().sum().sum()
        if missing > 0:
            issues.append(f"Found {missing} missing values")
                
        if np.any(np.isinf(data.select_dtypes(include=[np.number]))):
            issues.append("Found infinite values")
                
        constant_cols = data.columns[data.nunique() <= 1].tolist()
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
        
        return len(issues) == 0, issues

class RFEFeatureSelector(IFeatureSelector):
        
    def select(self, data: pd.DataFrame, target: str, n_features: int) -> List[str]:
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression
        
        X = data.drop(target, axis=1)
        y = data[target]
        
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        return selected

class LinearModelInterpreter(IModelInterpreter):
        
    def interpret(self, model, feature_names: List[str]) -> Dict[str, Any]:
        interpretation = {
            'coefficients': dict(zip(feature_names, model.coef_)),
            'intercept': float(model.intercept_),
            'feature_importance': dict(zip(feature_names, np.abs(model.coef_))),
            'most_important': feature_names[np.argmax(np.abs(model.coef_))] if len(feature_names) > 0 else None,
            'positive_impact': [feature_names[i] for i in range(len(feature_names)) if model.coef_[i] > 0],
            'negative_impact': [feature_names[i] for i in range(len(feature_names)) if model.coef_[i] < 0]
        }
        return interpretation


class SolidLinearRegressionPipeline:
        
    def __init__(self,
                 data_loader: IDataLoader,
                 data_validator: IDataValidator,
                 feature_engineer: IFeatureEngineer,
                 feature_selector: IFeatureSelector,
                 model_trainer: IRegressionTrainer,
                 model_evaluator: IRegressionEvaluator,
                 model_interpreter: IModelInterpreter):
        
        
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.model_interpreter = model_interpreter
                
        self.data = None
        self.model = None
        self.metrics = None
        self.interpretation = None
    
    def execute(self, data_source: str, target_column: str, n_features: int = 5):
        
        print(f"\n🚀 Executing SOLID Linear Regression Pipeline")
        print(f"Target: {target_column}, Max features: {n_features}")
                
        print("📥 Loading data...")
        self.data = self.data_loader.load(data_source)
        print(f"   Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
                
        print("🔍 Validating data...")
        is_valid, issues = self.data_validator.validate(self.data)
        if not is_valid:
            print(f"   Issues found: {issues}")
            # Handle issues or raise exception
        else:
            print("   ✓ Data validation passed")
        
        
        print("⚙️ Engineering features...")
        self.data = self.feature_engineer.engineer(self.data)
        print(f"   After engineering: {len(self.data.columns)} columns")
        
        
        print(" Selecting features...")
        selected_features = self.feature_selector.select(self.data, target_column, n_features)
        selected_features.append(target_column)
        self.data = self.data[selected_features]
        print(f"   Selected {len(selected_features)-1} features: {selected_features[:-1]}")
        
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        
        print("🏋️ Training model...")
        self.model = self.model_trainer.train(X, y)
        print("   ✓ Model trained")
        
        
        print(" Evaluating model...")
        self.metrics = self.model_evaluator.evaluate(self.model, X, y)
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        
        print(" Interpreting model...")
        self.interpretation = self.model_interpreter.interpret(
            self.model, 
            list(X.columns)
        )
        print(f"   Most important feature: {self.interpretation['most_important']}")
        print(f"   Intercept: {self.interpretation['intercept']:.4f}")
        
        return {
            'model': self.model,
            'metrics': self.metrics,
            'interpretation': self.interpretation,
            'selected_features': selected_features[:-1],
            'data_shape': self.data.shape
        }


print("\n📊 DEMONSTRATION OF SOLID LINEAR REGRESSION PIPELINE")


def create_sample_regression_data(n_samples=100, n_features=10):
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
        
    true_coef = np.random.randn(n_features)
        
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
        
    df.to_csv('sample_regression_data.csv', index=False)
    return df

print("\n1. Creating sample data...")
sample_data = create_sample_regression_data(200, 8)
print(f"   Created sample data with {len(sample_data)} samples")

print("\n2. Configuring SOLID pipeline components...")


pipeline = SolidLinearRegressionPipeline(
    data_loader=CSVDataLoaderDIP(),
    data_validator=DataQualityValidator(),
    feature_engineer=PolynomialFeatureEngineerDIP(),
    feature_selector=RFEFeatureSelector(),
    model_trainer=LinearRegressionTrainerDIP(),
    model_evaluator=ComprehensiveEvaluator(),
    model_interpreter=LinearModelInterpreter()
)

print("\n3. Pipeline demonstrates all SOLID principles:")
print("   ✅ SRP: Each component has single responsibility")
print("   ✅ OCP: Can extend any component without modification")
print("   ✅ LSP: Components are substitutable")
print("   ✅ ISP: Each interface is specific and focused")
print("   ✅ DIP: Pipeline depends on abstractions only")

print("\n4. Try running the pipeline:")
print("""

results = pipeline.execute(
    data_source='sample_regression_data.csv',
    target_column='target',
    n_features=5
)


print(f"Model R²: {results['metrics']['r2']:.4f}")
print(f"Selected features: {results['selected_features']}")
""")

# ============================================================================
# SUMMARY AND BENEFITS
# ============================================================================
print("\n\n" + "=" * 80)
print("SOLID PRINCIPLES BENEFITS FOR LINEAR REGRESSION")
print("=" * 80)

print("\n✅ SINGLE RESPONSIBILITY FOR REGRESSION:")
print("   - DataLoader: Only loads data")
print("   - FeatureEngineer: Only engineers features")
print("   - ModelTrainer: Only trains models")
print("   - ModelEvaluator: Only evaluates models")
print("   - ResultVisualizer: Only visualizes results")

print("\n✅ OPEN/CLOSED FOR REGRESSION:")
print("   - Add new regression type (ElasticNet) without modifying base")
print("   - Add new feature engineering technique without modifying pipeline")
print("   - Add new evaluation metric without modifying existing code")

print("\n✅ LISKOV SUBSTITUTION FOR REGRESSION:")
print("   - Any IRegressionTrainer can be substituted")
print("   - Any IDataLoader can be substituted")
print("   - Any IFeatureEngineer can be substituted")
print("   - Ensures consistent behavior across implementations")

print("\n✅ INTERFACE SEGREGATION FOR REGRESSION:")
print("   - IDataPipeline: Only data processing methods")
print("   - IFeatureEngineering: Only feature engineering methods")
print("   - IModelTraining: Only training methods")
print("   - IModelEvaluation: Only evaluation methods")
print("   - No forced implementation of unused methods")

print("\n✅ DEPENDENCY INVERSION FOR REGRESSION:")
print("   - Pipeline depends on IDataLoader, not CSVDataLoader")
print("   - Easy to switch from CSV to database")
print("   - Easy to switch from LinearRegression to RidgeRegression")
print("   - Easy to test with mock implementations")

print("\n" + "=" * 80)
print("SUCCESS! Linear Regression with SOLID Principles 🎯")
print("=" * 80)