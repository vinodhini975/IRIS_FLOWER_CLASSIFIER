from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target (0=setosa, 1=versicolor, 2=virginica)
df['target'] = iris.target

# Add target names for easy understanding
df['species'] = df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

# Print first 5 rows
print(df.tail())



import seaborn as sns
import matplotlib.pyplot as plt

# Draw pairplot (multi-feature scatter plot)
sns.pairplot(df, hue='species', markers=["o", "s", "D"])





from sklearn.model_selection import train_test_split

# Features (X) and Labels (y)
X = df[iris.feature_names]       # Input features
y = df['target']                 # Output label (0, 1, 2)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate accuracy
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Detailed performance report
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
# Predict a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input (setosa range)
predicted = model.predict(sample)
predicted_species = iris.target_names[predicted[0]]

print("🌸 Predicted Species for sample", sample[0], "→", predicted_species)

plt.show()