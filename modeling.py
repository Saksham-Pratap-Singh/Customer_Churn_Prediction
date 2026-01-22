# src/modeling.py
"""Machine learning model training and evaluation"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc, f1_score)
from imblearn.over_sampling import SMOTE
import logging
import joblib

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Train and evaluate churn prediction models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_test_split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        logger.info(f"Splitting data: test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        logger.info(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train, use_smote=True):
        """Handle class imbalance using SMOTE"""
        if not use_smote:
            return X_train, y_train
        
        logger.info("Applying SMOTE for class imbalance")
        smote = SMOTE(random_state=self.random_state, sampling_strategy=0.7)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Churn rate: {y_train_balanced.mean():.2%}")
        return X_train_balanced, y_train_balanced
    
    def train_logistic_regression(self, X_train, y_train, X_test=None, y_test=None):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression model")
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = model
        
        if X_test is not None and y_test is not None:
            self._evaluate_model(model, X_test, y_test, 'Logistic Regression')
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Train Random Forest model"""
        logger.info("Training Random Forest model")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        
        if X_test is not None and y_test is not None:
            self._evaluate_model(model, X_test, y_test, 'Random Forest')
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Train XGBoost model"""
        logger.info("Training XGBoost model")
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        self.models['XGBoost'] = model
        
        if X_test is not None and y_test is not None:
            self._evaluate_model(model, X_test, y_test, 'XGBoost')
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, use_smote=True):
        """Train all models and compare results"""
        logger.info("Training all models...")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train, use_smote)
        
        # Train models
        self.train_logistic_regression(X_train_balanced, y_train_balanced, X_test, y_test)
        self.train_random_forest(X_train_balanced, y_train_balanced, X_test, y_test)
        self.train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)
        
        # Find best model
        best_model_name = max(self.results, key=lambda k: self.results[k]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"\nBest Model: {best_model_name}")
        return self.results
    
    def get_results_dataframe(self):
        """Get results as DataFrame"""
        return pd.DataFrame(self.results).T
    
    def print_results_summary(self):
        """Print model comparison summary"""
        results_df = self.get_results_dataframe()
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(results_df.to_string())
        print("="*70)
        print(f"\nBest Model: {self.best_model_name} (ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f})")
    
    def get_feature_importance(self, model_name='XGBoost', top_n=15):
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importances = model.feature_importances_
            return importances
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return None
    
    def save_model(self, model_name, filepath):
        """Save model to disk"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
        return True
    
    def load_model(self, filepath, model_name=None):
        """Load model from disk"""
        model = joblib.load(filepath)
        
        if model_name:
            self.models[model_name] = model
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def predict(self, X, model_name=None):
        """Make predictions using specified or best model"""
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Train models first.")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from preprocessing import ChurnPreprocessor
    
    # Load and preprocess data
    preprocessor = ChurnPreprocessor()
    df = preprocessor.load_data('data/raw/telco_churn.csv')
    df = preprocessor.preprocess_pipeline(df)
    
    X, y = preprocessor.split_features_target(df)
    X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(X, y)
    X_train_scaled = preprocessor.scale_features(X_train, X_test)[0]
    
    # Train models
    trainer = ChurnModelTrainer()
    results = trainer.train_all_models(X_train_scaled, y_train, X_test, y_test)
    trainer.print_results_summary()