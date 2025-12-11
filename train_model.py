import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime

print("=" * 70)
print("🤖 LOCAL MODEL TRAINING - CARBON EMISSIONS PREDICTOR")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: Loading dataset...")
try:
    df = pd.read_csv('carbon_footprint_data.csv')
    print(f"✅ Successfully loaded {len(df)} records")
    print(f"✅ Columns: {list(df.columns)}")
except FileNotFoundError:
    print("❌ Error: 'carbon_footprint_data.csv' not found!")
    print("   Please make sure the CSV file is in the current directory")
    exit(1)

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
print("\n🔧 STEP 2: Preparing data...")

# Check for missing values
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"⚠️  Found {missing_values} missing values - removing them")
    df = df.dropna()

# Separate features (X) and target (y)
X = df[['electricity_kwh', 'gas_therms', 'distance_km', 'vehicle_type', 'shopping_carbon_kg']]
y = df['total_carbon_kg']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"✅ Feature columns: {list(X.columns)}")

# ============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\n🔀 STEP 3: Encoding categorical features...")

le = LabelEncoder()
X['vehicle_type'] = le.fit_transform(X['vehicle_type'])

print(f"✅ Vehicle type mapping:")
for class_name, encoded_value in zip(le.classes_, le.transform(le.classes_)):
    print(f"   {class_name:12s} → {encoded_value}")

# Save label encoder for later use
joblib.dump(le, 'label_encoder.pkl')
print(f"✅ Label encoder saved as 'label_encoder.pkl'")

# ============================================================================
# STEP 4: SPLIT DATA (80% train, 10% validation, 10% test)
# ============================================================================
print("\n📊 STEP 4: Splitting data (80/10/10)...")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42
)

print(f"✅ Training set:   {len(X_train):4d} rows (80%)")
print(f"✅ Validation set: {len(X_val):4d} rows (10%)")
print(f"✅ Test set:       {len(X_test):4d} rows (10%)")
print(f"✅ Total:          {len(df):4d} rows")

# ============================================================================
# STEP 5: SCALE FEATURES
# ============================================================================
print("\n📏 STEP 5: Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print(f"✅ Scaler saved as 'scaler.pkl'")

# ============================================================================
# STEP 6: TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n🤖 STEP 6: Training RandomForestRegressor model...")
print("   (This may take 30-60 seconds...)")

model = RandomForestRegressor(
    n_estimators=100,        # Number of trees
    max_depth=15,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    random_state=42,         # For reproducibility
    n_jobs=-1,              # Use all CPU cores
    verbose=0
)

model.fit(X_train_scaled, y_train)
print(f"✅ Model training complete!")

# ============================================================================
# STEP 7: EVALUATE ON TRAINING SET
# ============================================================================
print("\n📈 STEP 7: Evaluating on Training Set...")

y_train_pred = model.predict(X_train_scaled)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"   MAE (Train):    {train_mae:7.2f} kg CO2")
print(f"   RMSE (Train):   {train_rmse:7.2f} kg CO2")
print(f"   R² (Train):     {train_r2:7.4f}")

# ============================================================================
# STEP 8: EVALUATE ON VALIDATION SET
# ============================================================================
print("\n📈 STEP 8: Evaluating on Validation Set...")

y_val_pred = model.predict(X_val_scaled)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

print(f"   MAE (Val):      {val_mae:7.2f} kg CO2")
print(f"   RMSE (Val):     {val_rmse:7.2f} kg CO2")
print(f"   R² (Val):       {val_r2:7.4f}")

# ============================================================================
# STEP 9: EVALUATE ON TEST SET (FINAL PERFORMANCE)
# ============================================================================
print("\n📈 STEP 9: Evaluating on Test Set (FINAL PERFORMANCE)...")

y_test_pred = model.predict(X_test_scaled)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"   MAE (Test):     {test_mae:7.2f} kg CO2  ← Most Important")
print(f"   RMSE (Test):    {test_rmse:7.2f} kg CO2")
print(f"   R² (Test):      {test_r2:7.4f}       ← Model explains {test_r2*100:.1f}% of variance")

# ============================================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================================
print("\n⭐ STEP 10: Feature Importance Analysis...")

feature_names = ['electricity_kwh', 'gas_therms', 'distance_km', 'vehicle_type', 'shopping_carbon_kg']
importances = model.feature_importances_

print(f"\n   Which features matter most for predictions?")
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    percentage = importance * 100
    bar = "█" * int(percentage / 2)
    print(f"   {name:20s}: {importance:.4f} ({percentage:5.1f}%) {bar}")

# ============================================================================
# STEP 11: SAVE MODEL ARTIFACTS
# ============================================================================
print("\n💾 STEP 11: Saving model artifacts...")

# Create directory if it doesn't exist
os.makedirs('model_artifacts', exist_ok=True)

# Save the trained model
model_path = 'model_artifacts/carbon_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved: {model_path}")

# Save metrics to JSON file
metrics = {
    'model_type': 'RandomForestRegressor',
    'training_timestamp': datetime.now().isoformat(),
    'total_training_samples': len(df),
    'train_test_split': '80-10-10',
    'metrics': {
        'test_mae': round(float(test_mae), 2),
        'test_rmse': round(float(test_rmse), 2),
        'test_r2': round(float(test_r2), 4),
        'train_r2': round(float(train_r2), 4),
        'val_r2': round(float(val_r2), 4)
    },
    'feature_importance': {
        name: round(float(importance), 4) 
        for name, importance in zip(feature_names, importances)
    }
}

metrics_path = 'model_artifacts/metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved: {metrics_path}")

# ============================================================================
# STEP 12: SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ LOCAL TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
print(f"   Test R² Score:  {test_r2:.4f} (Excellent! >0.85 is great)")
print(f"   Test MAE:       {test_mae:.2f} kg CO2 (Average prediction error)")
print(f"   Test RMSE:      {test_rmse:.2f} kg CO2")

print(f"\n📦 FILES CREATED (ready for deployment):")
print(f"   ✅ model_artifacts/carbon_model.pkl")
print(f"   ✅ model_artifacts/metrics.json")
print(f"   ✅ label_encoder.pkl")
print(f"   ✅ scaler.pkl")

print(f"\n🚀 NEXT STEP:")
print(f"   Run: python upload_to_cloud.py")
print(f"   This will upload your model to Google Cloud Storage")

print("\n" + "=" * 70)