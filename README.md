# INTERNORBIT-task two
# iris_flower_classification.py

# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load dataset
df = sns.load_dataset('iris')
print("Dataset Head:\n", df.head())

# Step 3: Data Exploration
print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['species'].value_counts())

# Step 4: Data Visualization
sns.pairplot(df, hue='species')
plt.title("Iris Flower Feature Pair Plot")
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Split the dataset
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 10: Predict a sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nPredicted Species for sample", sample, "is:", prediction[0])
