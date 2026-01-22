# main.py - Complete ML Pipeline Execution

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import ChurnPreprocessor
from modeling import ChurnModelTrainer
import config

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Execute the complete ML pipeline"""
    
    logger.info("="*70)
    logger.info("CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    logger.info("="*70)
    
    # ============ STEP 1: DATA LOADING & EXPLORATION ============
    logger.info("\n[STEP 1] Loading and Exploring Data")
    logger.info("-" * 70)
    
    preprocessor = ChurnPreprocessor()
    
    # Check if data exists
    if not config.RAW_DATA_PATH.exists():
        logger.error(f"Data file not found at {config.RAW_DATA_PATH}")
        logger.info("Please download the dataset from Kaggle and place it in data/raw/")
        return
    
    df = preprocessor.load_data(str(config.RAW_DATA_PATH))
    preprocessor.explore_data(df, target_col='Churn')
    
    # ============ STEP 2: DATA PREPROCESSING ============
    logger.info("\n[STEP 2] Data Preprocessing")
    logger.info("-" * 70)
    
    # Data cleaning
    df_processed = df.copy()
    df_processed = preprocessor.remove_duplicates(df_processed)
    df_processed = preprocessor.handle_missing_values(df_processed)
    
    # Rename columns to lowercase for consistency
    df_processed.columns = df_processed.columns.str.lower()
    
    # Encode target variable
    df_processed = preprocessor.encode_target(df_processed, target_col='churn')
    
    # Encode categorical features
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    df_processed = preprocessor.encode_categorical_features(df_processed, categorical_features)
    
    # Feature engineering
    df_processed = preprocessor.create_features(df_processed)
    
    logger.info(f"Processed data shape: {df_processed.shape}")
    
    # ============ STEP 3: FEATURE PREPARATION ============
    logger.info("\n[STEP 3] Feature Preparation")
    logger.info("-" * 70)
    
    X, y = preprocessor.split_features_target(df_processed, target_col='churn')
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribution - No: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%), "
                f"Yes: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # ============ STEP 4: TRAIN-TEST SPLIT ============
    logger.info("\n[STEP 4] Train-Test Split")
    logger.info("-" * 70)
    
    trainer = ChurnModelTrainer(random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = trainer.train_test_split_data(
        X, y, test_size=config.TEST_SIZE
    )
    
    # ============ STEP 5: FEATURE SCALING ============
    logger.info("\n[STEP 5] Feature Scaling")
    logger.info("-" * 70)
    
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    logger.info("Features scaled using StandardScaler")
    
    # ============ STEP 6: HANDLE CLASS IMBALANCE ============
    logger.info("\n[STEP 6] Handling Class Imbalance")
    logger.info("-" * 70)
    
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(
        X_train_scaled, y_train, use_smote=config.USE_SMOTE
    )
    
    # ============ STEP 7: MODEL TRAINING ============
    logger.info("\n[STEP 7] Training Multiple Models")
    logger.info("-" * 70)
    
    results = trainer.train_all_models(
        X_train_balanced, y_train_balanced, X_test_scaled, y_test,
        use_smote=False  # Already handled above
    )
    
    # Print summary
    trainer.print_results_summary()
    
    # ============ STEP 8: FEATURE IMPORTANCE ============
    logger.info("\n[STEP 8] Feature Importance Analysis")
    logger.info("-" * 70)
    
    xgb_importance = trainer.get_feature_importance('XGBoost')
    if xgb_importance is not None:
        feature_importance_df = pd.DataFrame({
            'feature': X_train_scaled.columns,
            'importance': xgb_importance
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        logger.info(feature_importance_df.head(10).to_string(index=False))
    
    # ============ STEP 9: SAVE MODELS ============
    logger.info("\n[STEP 9] Saving Models")
    logger.info("-" * 70)
    
    trainer.save_model('XGBoost', str(config.BEST_MODEL_PATH))
    trainer.save_model('Logistic Regression', str(config.LOGISTIC_MODEL_PATH))
    trainer.save_model('Random Forest', str(config.RANDOM_FOREST_MODEL_PATH))
    preprocessor.save_scaler(str(config.SCALER_PATH))
    
    logger.info("All models saved successfully!")
    
    # ============ STEP 10: SAVE PROCESSED DATA ============
    logger.info("\n[STEP 10] Saving Processed Data")
    logger.info("-" * 70)
    
    df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed data saved to {config.PROCESSED_DATA_PATH}")
    
    # ============ COMPLETION ============
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    
    logger.info("\n📊 SUMMARY:")
    logger.info(f"  • Best Model: {trainer.best_model_name}")
    logger.info(f"  • Best ROC-AUC: {results[trainer.best_model_name]['roc_auc']:.4f}")
    logger.info(f"  • Recall: {results[trainer.best_model_name]['recall']:.4f}")
    logger.info(f"  • Precision: {results[trainer.best_model_name]['precision']:.4f}")
    
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("  1. Run Streamlit app: streamlit run app.py")
    logger.info("  2. Review EDA visualizations in outputs/")
    logger.info("  3. Push to GitHub for portfolio")
    logger.info("  4. Deploy to Streamlit Cloud")
    
    # Return results for further analysis
    return trainer, preprocessor, X_test_scaled, y_test


if __name__ == "__main__":
    try:
        trainer, preprocessor, X_test, y_test = main()
        
        # Example: Make predictions on test set
        logger.info("\n" + "="*70)
        logger.info("EXAMPLE PREDICTIONS")
        logger.info("="*70)
        
        predictions, probabilities = trainer.predict(X_test.iloc[:5])
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "Will Churn ⚠️" if pred == 1 else "Stable ✅"
            logger.info(f"Customer {i+1}: {status} (Probability: {prob:.2%})")
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)