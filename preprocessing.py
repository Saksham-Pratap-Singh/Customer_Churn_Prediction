# src/preprocessing.py
"""Data preprocessing and feature engineering utilities"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class ChurnPreprocessor:
    """Handle data preprocessing and feature engineering for churn prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
    
    def load_data(self, filepath):
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def explore_data(self, df, target_col='churn'):
        """Print basic data exploration statistics"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        
        if target_col in df.columns:
            print(f"\nTarget Variable Distribution:")
            print(df[target_col].value_counts(normalize=True) * 100)
        
        print(f"\nBasic Statistics:")
        print(df.describe())
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Numerical columns - use mean/median
        if len(numerical_cols) > 0:
            imputer = SimpleImputer(strategy=strategy)
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        # Categorical columns - use most frequent
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        
        logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df
    
    def encode_target(self, df, target_col='churn'):
        """Encode target variable (binary: Yes/No → 1/0)"""
        logger.info(f"Encoding target variable: {target_col}")
        
        if df[target_col].dtype == 'object':
            target_encoder = LabelEncoder()
            df[target_col] = target_encoder.fit_transform(df[target_col])
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """Encode categorical features using LabelEncoder"""
        logger.info("Encoding categorical features")
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def create_features(self, df):
        """Create engineered features from existing ones"""
        logger.info("Creating engineered features")
        
        # Ratio of monthly to total charges
        df['avg_monthly_to_total'] = df['monthly_charges'] / (df['total_charges'] + 1)
        
        # High monthly charge flag
        if 'monthly_charges' in df.columns:
            df['high_monthly_charge'] = (df['monthly_charges'] > 
                                         df['monthly_charges'].median()).astype(int)
        
        # New customer flag (tenure < 6 months)
        if 'tenure' in df.columns:
            df['new_customer'] = (df['tenure'] < 6).astype(int)
            df['long_term_customer'] = (df['tenure'] > 24).astype(int)
            
            # Tenure groups
            df['tenure_group'] = pd.cut(df['tenure'], 
                                       bins=[0, 6, 12, 24, 72],
                                       labels=[1, 2, 3, 4]).astype(int)
        
        # Service count (sum of binary service columns)
        service_cols = ['online_security', 'online_backup', 'device_protection', 
                       'tech_support', 'streaming_tv', 'streaming_movies']
        service_cols = [col for col in service_cols if col in df.columns]
        if service_cols:
            df['service_count'] = df[service_cols].sum(axis=1)
        
        logger.info(f"Created {len([col for col in df.columns if 'avg_' in col or col in 
                   ['high_monthly_charge', 'new_customer', 'long_term_customer', 
                    'tenure_group', 'service_count']])} engineered features")
        
        return df
    
    def split_features_target(self, df, target_col='churn'):
        """Separate features and target variable"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_names = X.columns
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Features split: {X.shape[1]} features, Target shape: {y.shape}")
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features using StandardScaler"""
        logger.info("Scaling features")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def preprocess_pipeline(self, df, target_col='churn', categorical_cols=None):
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # Explore data
        self.explore_data(df, target_col)
        
        # Clean data
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        
        # Encode
        df = self.encode_target(df, target_col)
        df = self.encode_categorical_features(df, categorical_cols)
        
        # Feature engineering
        df = self.create_features(df)
        
        logger.info("Preprocessing complete")
        return df
    
    def save_scaler(self, filepath):
        """Save fitted scaler to disk"""
        import joblib
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath):
        """Load scaler from disk"""
        import joblib
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = ChurnPreprocessor()
    df = preprocessor.load_data('data/raw/telco_churn.csv')
    df_processed = preprocessor.preprocess_pipeline(df)
    print("\nProcessing completed successfully!")
    print(f"Processed data shape: {df_processed.shape}")