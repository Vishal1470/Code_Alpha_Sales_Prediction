# ======================================================
# TRAIN MODEL SCRIPT (with graph)
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle, os

# 1️⃣ Load dataset
df = pd.read_csv('dataset/advertising.csv')

# 2️⃣ Data overview
print("Dataset Head:\n", df.head())
print("\nColumns:", df.columns.tolist())

# 3️⃣ Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 4️⃣ Split features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 7️⃣ Evaluate model
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

# 8️⃣ Plot actual vs predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# 9️⃣ Save model
os.makedirs('model', exist_ok=True)
with open('model/sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully in 'model/sales_model.pkl'")

