import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

# ==========================================
# 1. INPUT: Load Data
# ==========================================
iris = load_iris()
# We select the first two features (Sepal Length & Sepal Width) to easily plot them in 2D
X = iris.data[:, :2] 
y = iris.target

print("--- 1. INPUT PHASE COMPLETE ---")

# ==========================================
# 2. PROCESS: Split, Scale, and Train
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

print("--- 2. PROCESS PHASE COMPLETE ---")

# ==========================================
# 3. OUTPUT: Evaluate and Plot Graph
# ==========================================
predictions = model.predict(X_test_scaled)
print("\n--- 3. OUTPUT PHASE (EVALUATION) ---")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# --- CREATE THE VISUAL GRAPH ---
plt.figure(figsize=(8, 6))

# Plot the training data points
sns.scatterplot(
    x=X_train_scaled[:, 0], 
    y=X_train_scaled[:, 1], 
    hue=[iris.target_names[i] for i in y_train],
    palette='viridis', 
    s=70, 
    edgecolor='k'
)

plt.title('AI Supervised Learning: Iris Data Classification')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Sepal Width (Standardized)')
plt.legend(title='Flower Class')
plt.grid(True, linestyle='--', alpha=0.6)

print("\n[INFO] Displaying the classification graph... Close the window to finish.")
plt.show()