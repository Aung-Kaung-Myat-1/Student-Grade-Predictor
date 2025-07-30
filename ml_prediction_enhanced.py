import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load dataset
df = pd.read_csv('Data/StudentsPerformance.csv')

print("=" * 60)
print("ENHANCED MACHINE LEARNING: MATH SCORE PREDICTION")
print("=" * 60)

# Original features (numeric only)
X_original = df[['reading score', 'writing score']]
y = df['math score']

# Enhanced features with categorical variables
categorical_features = ['gender', 'parental level of education', 'lunch', 'test preparation course']
numeric_features = ['reading score', 'writing score']

# Create one-hot encoder
encoder = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])

# Get feature names for categorical variables
categorical_feature_names = []
for i, feature in enumerate(categorical_features):
    categories = encoder.categories_[i][1:]  # Drop first category
    for category in categories:
        categorical_feature_names.append(f"{feature}_{category}")

# Combine numeric and encoded categorical features
X_enhanced = np.hstack([df[numeric_features].values, categorical_encoded])

print(f"Original dataset shape: {df.shape}")
print(f"Original features (X): {X_original.shape}")
print(f"Enhanced features (X): {X_enhanced.shape}")
print(f"Target (y): {y.shape}")

# Split data for both models
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)
X_train_enh, X_test_enh, _, _ = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train_orig.shape[0]} samples")
print(f"Test set size: {X_test_orig.shape[0]} samples")

# Train original model
print("\n" + "=" * 60)
print("ORIGINAL MODEL (NUMERIC FEATURES ONLY)")
print("=" * 60)

model_original = DecisionTreeRegressor(random_state=42)
model_original.fit(X_train_orig, y_train)

# Predict and evaluate original model
y_pred_orig = model_original.predict(X_test_orig)
mse_orig = mean_squared_error(y_test, y_pred_orig)
rmse_orig = np.sqrt(mse_orig)
mae_orig = mean_absolute_error(y_test, y_pred_orig)
r2_orig = r2_score(y_test, y_pred_orig)

print(f"Mean Squared Error (MSE): {mse_orig:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_orig:.2f}")
print(f"Mean Absolute Error (MAE): {mae_orig:.2f}")
print(f"R-squared (R²): {r2_orig:.3f}")

# Feature importance for original model
feature_importance_orig = model_original.feature_importances_
print(f"\nFeature Importance (Original):")
for feature, importance in zip(X_original.columns, feature_importance_orig):
    print(f"  {feature}: {importance:.3f}")

# Train enhanced model
print("\n" + "=" * 60)
print("ENHANCED MODEL (WITH CATEGORICAL FEATURES)")
print("=" * 60)

model_enhanced = DecisionTreeRegressor(random_state=42)
model_enhanced.fit(X_train_enh, y_train)

# Predict and evaluate enhanced model
y_pred_enh = model_enhanced.predict(X_test_enh)
mse_enh = mean_squared_error(y_test, y_pred_enh)
rmse_enh = np.sqrt(mse_enh)
mae_enh = mean_absolute_error(y_test, y_pred_enh)
r2_enh = r2_score(y_test, y_pred_enh)

print(f"Mean Squared Error (MSE): {mse_enh:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_enh:.2f}")
print(f"Mean Absolute Error (MAE): {mae_enh:.2f}")
print(f"R-squared (R²): {r2_enh:.3f}")

# Feature importance for enhanced model
feature_importance_enh = model_enhanced.feature_importances_
all_feature_names = list(numeric_features) + categorical_feature_names

print(f"\nFeature Importance (Enhanced):")
for feature, importance in zip(all_feature_names, feature_importance_enh):
    print(f"  {feature}: {importance:.3f}")

# Model comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Original Model:")
print(f"  MSE: {mse_orig:.2f}")
print(f"  RMSE: {rmse_orig:.2f}")
print(f"  MAE: {mae_orig:.2f}")
print(f"  R²: {r2_orig:.3f}")

print(f"\nEnhanced Model:")
print(f"  MSE: {mse_enh:.2f}")
print(f"  RMSE: {rmse_enh:.2f}")
print(f"  MAE: {mae_enh:.2f}")
print(f"  R²: {r2_enh:.3f}")

# Calculate improvements
mse_improvement = ((mse_orig - mse_enh) / mse_orig) * 100
r2_improvement = ((r2_enh - r2_orig) / r2_orig) * 100

print(f"\nImprovements:")
print(f"  MSE reduction: {mse_improvement:.1f}%")
print(f"  R² improvement: {r2_improvement:.1f}%")

# Create comparison scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Original model scatter plot
ax1.scatter(y_test, y_pred_orig, alpha=0.6, color='skyblue', s=50, edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred_orig.min(), y_pred_enh.min())
max_val = max(y_test.max(), y_pred_orig.max(), y_pred_enh.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Math Score')
ax1.set_ylabel('Predicted Math Score')
ax1.set_title(f'Original Model\nR² = {r2_orig:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Enhanced model scatter plot
ax2.scatter(y_test, y_pred_enh, alpha=0.6, color='lightgreen', s=50, edgecolors='black', linewidth=0.5)
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Math Score')
ax2.set_ylabel('Predicted Math Score')
ax2.set_title(f'Enhanced Model\nR² = {r2_enh:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Model insights
print(f"\n" + "=" * 60)
print("MODEL INSIGHTS")
print("=" * 60)
print(f"• Original model uses only reading and writing scores")
print(f"• Enhanced model includes {len(categorical_feature_names)} additional categorical features")
print(f"• Enhanced model features: {', '.join(categorical_feature_names[:5])}...")
print(f"• Total enhanced features: {len(all_feature_names)}")
print(f"• The enhanced model {'improves' if mse_enh < mse_orig else 'does not improve'} performance") 