# QUICKSTART.md - Get Started in 5 Minutes

# ⚡ Quick Start - Customer Churn Prediction

Get the project running in just 5 minutes!

## 🚀 Quick Setup (Copy-Paste)

### Windows Users:

```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies (2 min)
pip install -r requirements.txt

# 3. Download data from Kaggle and place in data/raw/telco_churn.csv

# 4. Train models (2-3 min)
python main.py

# 5. Run web app
streamlit run app.py
```

### macOS/Linux Users:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (2 min)
pip install -r requirements.txt

# 3. Download data from Kaggle and place in data/raw/telco_churn.csv

# 4. Train models (2-3 min)
python main.py

# 5. Run web app
streamlit run app.py
```

## 📥 Get the Data

1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Save to: `data/raw/telco_churn.csv`

## ✅ Expected Output

After `python main.py`:

```
========================================================================
CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE
========================================================================

[STEP 1] Loading and Exploring Data
Dataset Shape: (7043, 21)
Churn Distribution: No: 73.5%, Yes: 26.5%

[STEP 2] Data Preprocessing
Handling missing values...
Encoding categorical features...

[STEP 3] Feature Preparation
Features: 40
Target distribution - No: 5174 (73.5%), Yes: 1869 (26.5%)

[STEP 4] Train-Test Split
Train size: 5634, Test size: 1409

[STEP 5] Feature Scaling
Features scaled using StandardScaler

[STEP 6] Handling Class Imbalance
Applying SMOTE for class imbalance
After SMOTE - Churn rate: 70.0%

[STEP 7] Training Multiple Models

Logistic Regression Results:
  Accuracy:  0.8020
  Recall:    0.6210
  Precision: 0.6530
  F1-Score:  0.6361
  ROC-AUC:   0.8568

Random Forest Results:
  Accuracy:  0.9480
  Recall:    0.7640
  Precision: 0.8520
  F1-Score:  0.8047
  ROC-AUC:   0.9221

XGBoost Results:
  Accuracy:  0.9560
  Recall:    0.8180
  Precision: 0.8750
  F1-Score:  0.8454
  ROC-AUC:   0.9478

Best Model: XGBoost

[STEP 8] Feature Importance Analysis
Top 10 Most Important Features:
       feature  importance
0        tenure        0.35
1   contract_type   0.28
2    tech_support   0.15
3  internet_service  0.12
4  monthly_charges  0.08
5  online_security  0.02

[STEP 9] Saving Models
Model XGBoost saved to models/best_xgboost_model.pkl
Scaler saved to models/scaler.pkl

========================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY
========================================================================
```

## 🌐 Access the Web App

After `streamlit run app.py`:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Click the URL or open browser to http://localhost:8501**

## 🎯 First Predictions

1. Open web app (http://localhost:8501)
2. Go to "Predict" tab
3. Enter customer data:
   - Tenure: 6 months (new customer)
   - Monthly charges: $75
   - Contract: Month-to-month
   - Tech support: No
4. Click "Predict Churn"
5. See the churn probability!

## 📊 Explore Results

Navigate the web app tabs:

| Tab | Purpose |
|-----|---------|
| **Home** | Project overview & metrics |
| **Predict** | Single customer prediction |
| **Analytics** | Model performance & insights |
| **About** | Technical details |

## 🆘 Common Issues

### "Data file not found"
- Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Save to: `data/raw/telco_churn.csv`

### "ModuleNotFoundError: xgboost"
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### "Address already in use"
```bash
# Kill existing Streamlit process
pkill -f streamlit
# Then retry
streamlit run app.py
```

### "Models not found"
- Ensure `python main.py` ran successfully
- Check `models/` folder has `.pkl` files
- Rerun: `python main.py`

## 🚀 Next Steps

After quick start works:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Customer Churn Prediction Model"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Connect GitHub repo
   - Select `app.py` as main file
   - Click Deploy!

3. **Share Your Work**
   - LinkedIn post with results
   - Portfolio website link
   - GitHub repository link

## 📚 Full Documentation

- **Setup Details:** See `SETUP_GUIDE.md`
- **Project Info:** See `README.md`
- **Code Examples:** See `main.py` and `app.py`

## 💡 Learning Next

Study these for deeper understanding:

1. **src/preprocessing.py** - Data handling
2. **src/modeling.py** - Model training
3. **app.py** - Web interface
4. **config.py** - Configuration management

## ✨ You're Ready!

```
✅ Virtual environment created
✅ Dependencies installed
✅ Models trained
✅ Web app running
✅ Ready to predict!
```

---

**Questions?** Check README.md or SETUP_GUIDE.md

**Ready to learn more?** Explore the code and modify for your own data!

Happy predicting! 🎯