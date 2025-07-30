import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load dataset
df = pd.read_csv('Data/StudentsPerformance.csv')

print("=" * 50)
print("MACHINE LEARNING: MATH SCORE PREDICTION")
print("=" * 50)

# Features and target
X = df[['reading score', 'writing score']]
y = df['math score']

print(f"Dataset shape: {df.shape}")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Initialize and train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.3f}")

# Feature importance
feature_importance = model.feature_importances_
print(f"\nFeature Importance:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"  {feature}: {importance:.3f}")

# Sample predictions
print(f"\nSample Predictions (first 10 test samples):")
print("Actual vs Predicted:")
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"  Sample {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}, Difference={abs(actual-predicted):.1f}")

# Create scatter plot comparing actual vs predicted values
plt.figure(figsize=(10, 8))

# Scatter plot of actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6, color='skyblue', s=50, edgecolors='black', linewidth=0.5)

# Add diagonal line for perfect prediction
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

# Customize the plot
plt.xlabel('Actual Math Score', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Math Score', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Math Scores\nDecision Tree Regressor', fontsize=14, fontweight='bold')

# Add R² value as text on the plot
plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         fontsize=12, verticalalignment='top')

plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Set axis limits to be the same
plt.xlim(min_val - 5, max_val + 5)
plt.ylim(min_val - 5, max_val + 5)

plt.tight_layout()
plt.savefig('actual_vs_predicted_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# Model insights
print(f"\n" + "=" * 50)
print("MODEL INSIGHTS")
print("=" * 50)
print(f"• The model uses reading and writing scores to predict math scores")
print(f"• R² of {r2:.3f} indicates the model explains {r2*100:.1f}% of the variance")
print(f"• Average prediction error: {mae:.1f} points")
print(f"• Root mean squared error: {rmse:.1f} points")
print(f"• Points closer to the red diagonal line indicate better predictions") 