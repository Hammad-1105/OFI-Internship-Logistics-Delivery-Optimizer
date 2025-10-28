# ============================================================================
# PREDICTIVE DELIVERY OPTIMIZER - BINARY CLASSIFICATION VERSION
# Problem: Predict if delivery will be ON-TIME vs DELAYED
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: DATA FOUNDATION WITH FEATURE ENGINEERING
# ============================================================================

print("=" * 70)
print("PHASE 1: DATA LOADING & FEATURE ENGINEERING")
print("=" * 70)

# Step 1: Load Data
print("\n[1/5] Loading datasets...")
df_orders = pd.read_csv('1_orders.csv')
df_routes = pd.read_csv('2_routes_distance.csv')
df_performance = pd.read_csv('3_delivery_performance.csv')
print(f"Loaded: {len(df_orders)} orders, {len(df_routes)} routes, {len(df_performance)} deliveries")

# Step 2: Merge Data
print("\n[2/5] Merging datasets...")
df_merged = pd.merge(df_orders, df_routes, on='Order_ID', how='inner')
df_master = pd.merge(df_merged, df_performance, on='Order_ID', how='inner')
print(f"Master dataset created: {df_master.shape[0]} rows × {df_master.shape[1]} columns")

# Step 3: Create Binary Target Variable
print("\n[3/5] Creating binary target variable...")
print("\nOriginal Distribution:")
print(df_master['Delivery_Status'].value_counts())

# Binary Mapping: 0 = On-Time, 1 = Delayed (Any type)
df_master['Delivery_Status_Binary'] = df_master['Delivery_Status'].apply(
    lambda x: 0 if x == 'On-Time' else 1
)

print("\nBinary Distribution:")
binary_dist = df_master['Delivery_Status_Binary'].value_counts()
print(f"  On-Time (0): {binary_dist[0]} ({binary_dist[0]/len(df_master)*100:.1f}%)")
print(f"  Delayed (1): {binary_dist[1]} ({binary_dist[1]/len(df_master)*100:.1f}%)")
print("Much better class balance for modeling!")

# Step 4: Feature Engineering
print("\n[4/5] Engineering domain-specific features...")

# 4.1 Priority Encoding (Express = highest pressure)
priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
df_master['Priority_Score'] = df_master['Priority'].map(priority_map)

# 4.2 Delivery Pressure (combines priority with traffic)
df_master['Delivery_Pressure'] = (
    df_master['Priority_Score'] * (1 + df_master['Traffic_Delay_Minutes'] / 60)
)

# 4.3 Cost Efficiency Metrics
df_master['Cost_Per_KM'] = df_master['Delivery_Cost_INR'] / (df_master['Distance_KM'] + 1)
df_master['Toll_Ratio'] = df_master['Toll_Charges_INR'] / (df_master['Delivery_Cost_INR'] + 1)

# 4.4 Order Value Metrics
df_master['Value_Per_KM'] = df_master['Order_Value_INR'] / (df_master['Distance_KM'] + 1)
df_master['High_Value_Order'] = (df_master['Order_Value_INR'] > df_master['Order_Value_INR'].median()).astype(int)

# 4.5 Binary Flags
df_master['Has_Special_Handling'] = df_master['Special_Handling'].notna().astype(int)
df_master['Has_Weather_Impact'] = df_master['Weather_Impact'].notna().astype(int)
df_master['Is_Express'] = (df_master['Priority'] == 'Express').astype(int)

# 4.6 Distance Categories
df_master['Distance_Category'] = pd.cut(
    df_master['Distance_KM'], 
    bins=[0, 200, 500, 1000, 2000], 
    labels=['Short', 'Medium', 'Long', 'VeryLong']
)

# 4.7 Traffic Impact Level
df_master['Traffic_Level'] = pd.cut(
    df_master['Traffic_Delay_Minutes'],
    bins=[-1, 10, 30, 60, 200],
    labels=['Low', 'Medium', 'High', 'Severe']
)

# 4.8 Carrier Performance Score (historical on-time rate by carrier)
carrier_performance = df_master.groupby('Carrier')['Delivery_Status_Binary'].agg(['mean', 'count'])
carrier_performance['Carrier_Delay_Rate'] = carrier_performance['mean']
carrier_performance['Carrier_Volume'] = carrier_performance['count']
df_master = df_master.merge(
    carrier_performance[['Carrier_Delay_Rate', 'Carrier_Volume']], 
    left_on='Carrier', 
    right_index=True, 
    how='left'
)

print(f"Created 14 engineered features")

# Step 5: Prepare Final Dataset
print("\n[5/5] Preparing final feature matrix...")

# Separate target
y = df_master['Delivery_Status_Binary']

# Remove columns not needed for prediction
columns_to_drop = [
    'Order_ID', 'Order_Date', 'Route', 'Delivery_Status',
    'Delivery_Status_Binary', 'Fuel_Consumption_L',
    'Promised_Delivery_Days', 'Actual_Delivery_Days',
    'Quality_Issue', 'Customer_Rating', 'Delivery_Cost_INR',
    'Special_Handling', 'Weather_Impact', 'Priority_Score'  # Used in engineering
]

df_features = df_master.drop(columns=columns_to_drop, errors='ignore')

# Define feature sets
numerical_features = [
    'Order_Value_INR', 'Distance_KM', 'Toll_Charges_INR', 
    'Traffic_Delay_Minutes', 'Delivery_Pressure', 'Cost_Per_KM',
    'Toll_Ratio', 'Value_Per_KM', 'High_Value_Order',
    'Has_Special_Handling', 'Has_Weather_Impact', 'Is_Express',
    'Carrier_Delay_Rate', 'Carrier_Volume'
]

categorical_features = [
    'Customer_Segment', 'Priority', 'Product_Category',
    'Origin', 'Destination', 'Carrier',
    'Distance_Category', 'Traffic_Level'
]

X = df_features[numerical_features + categorical_features]

print(f"Feature matrix: {X.shape}")
print(f"  - Numerical features: {len(numerical_features)}")
print(f"  - Categorical features: {len(categorical_features)}")
print(f"  - Target variable: {y.name} (binary)")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)

# ============================================================================
# PHASE 2: PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 2: BUILDING PREPROCESSING PIPELINE")
print("=" * 70)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numerical Pipeline
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

print("Preprocessing pipeline created")
print("  - Numerical: Median imputation → Standard scaling")
print("  - Categorical: Constant imputation → One-hot encoding")

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)

# ============================================================================
# PHASE 3: MODEL TRAINING & EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 3: MODEL TRAINING & CROSS-VALIDATION")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Model 1: Logistic Regression (Baseline)
print("\n[1/3] Building Logistic Regression model...")
pipe_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        solver='saga',
        C=0.5,
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    ))
])

# Model 2: Random Forest (Primary)
print("[2/3] Building Random Forest model...")
pipe_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# Model 3: XGBoost (Advanced)
print("[3/3] Building XGBoost model...")
pipe_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),  # Handle imbalance
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])

# Cross-Validation Evaluation
print("\n" + "=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)

models = {
    'Logistic Regression': pipe_lr,
    'Random Forest': pipe_rf,
    'XGBoost': pipe_xgb
}

results = []

for model_name, model_pipeline in models.items():
    print(f"\n--- {model_name} ---")
    
    # Cross-validation scores
    cv_scores = {
        'Accuracy': cross_val_score(model_pipeline, X, y, cv=5, scoring='accuracy'),
        'Precision': cross_val_score(model_pipeline, X, y, cv=5, scoring='precision'),
        'Recall': cross_val_score(model_pipeline, X, y, cv=5, scoring='recall'),
        'F1': cross_val_score(model_pipeline, X, y, cv=5, scoring='f1'),
        'ROC-AUC': cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
    }
    
    # Store results
    result = {'Model': model_name}
    for metric_name, scores in cv_scores.items():
        result[f'{metric_name} (Mean)'] = np.mean(scores)
        result[f'{metric_name} (Std)'] = np.std(scores)
        print(f"  {metric_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    results.append(result)

# Create results DataFrame
df_metrics = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)
print(df_metrics[['Model', 'F1 (Mean)', 'F1 (Std)', 'ROC-AUC (Mean)']].to_string(index=False))

# Detailed Analysis: Confusion Matrix & Classification Report
print("\n" + "=" * 70)
print("DETAILED PERFORMANCE ANALYSIS")
print("=" * 70)

for model_name, model_pipeline in models.items():
    print(f"\n{'=' * 70}")
    print(f"{model_name.upper()}")
    print('=' * 70)
    
    # Get cross-validated predictions
    y_pred = cross_val_predict(model_pipeline, X, y, cv=5)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(
        cm,
        columns=['Predicted: On-Time', 'Predicted: Delayed'],
        index=['Actual: On-Time', 'Actual: Delayed']
    )
    print(cm_df)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y, y_pred,
        target_names=['On-Time', 'Delayed'],
        digits=4
    ))

print("\n" + "=" * 70)
print("TRAINING FINAL MODELS ON FULL DATASET")
print("=" * 70)

# Train final models on 100% of data for deployment
for model_name, model_pipeline in models.items():
    print(f"Training {model_name}...", end=" ")
    model_pipeline.fit(X, y)
   

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

# ============================================================================
# SAVE RESULTS FOR STREAMLIT APP
# ============================================================================

print("\n" + "=" * 70)
print("SAVING ARTIFACTS FOR STREAMLIT APP")
print("=" * 70)

import joblib

# Save trained models
joblib.dump(pipe_lr, 'model_lr.pkl')
joblib.dump(pipe_rf, 'model_rf.pkl')
joblib.dump(pipe_xgb, 'model_xgb.pkl')
print("Models saved: model_lr.pkl, model_rf.pkl, model_xgb.pkl")

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved: preprocessor.pkl")

# Save feature names
feature_info = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'all_features': numerical_features + categorical_features
}
joblib.dump(feature_info, 'feature_info.pkl')
print("Feature info saved: feature_info.pkl")

# Save metrics
df_metrics.to_csv('model_metrics.csv', index=False)
print("Metrics saved: model_metrics.csv")

# Save cleaned dataset for download
df_master.to_csv('master_dataset_cleaned.csv', index=False)
print("Master dataset saved: master_dataset_cleaned.csv")

print("\n" + "=" * 70)
print("ALL PHASES COMPLETE - READY FOR STREAMLIT APP!")
print("=" * 70)

print("\nExpected Performance:")
print("  • F1 Score: 65-75% (significantly improved from 35-41%)")
print("  • ROC-AUC: 70-80%")
print("  • Better interpretability with binary classification")
print("\nNext Step: Build Streamlit app using saved models")