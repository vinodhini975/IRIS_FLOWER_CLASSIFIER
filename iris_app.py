import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter sepal and petal dimensions to predict the flower species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
sample = [[sepal_length, sepal_width, petal_length, petal_width]]
# Predict class and get probability scores
prediction = model.predict(sample)
probabilities = model.predict_proba(sample)

species = iris.target_names[prediction[0]]
confidence = probabilities[0][prediction[0]] * 100  # Convert to %

# Show result
st.success(f"🌼 Predicted Species: **{species.capitalize()}** with {confidence:.2f}% confidence")
# Show flower image based on prediction
if species == "setosa":
    st.image("setosa.png", caption="Iris Setosa", use_column_width=True)
elif species == "versicolor":
    st.image("versicolor.png", caption="Iris Versicolor", use_column_width=True)
elif species == "virginica":
    st.image("virginica.png", caption="Iris Virginica", use_column_width=True)




# Show probabilities for all classes
st.subheader("📊 Prediction Confidence Breakdown")
for i, prob in enumerate(probabilities[0]):
    # Bar chart for prediction probabilities
    fig, ax = plt.subplots()
    species_names = [name.capitalize() for name in iris.target_names]
    ax.bar(species_names, probabilities[0] * 100, color=['green', 'orange', 'purple'])
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Prediction Confidence by Species')
    st.pyplot(fig)

    st.write(f"{iris.target_names[i].capitalize()}: {prob * 100:.2f}%")
# Prepare prediction report text
report_text = f"""
Iris Flower Prediction Report

Input Features:
- Sepal Length: {sepal_length} cm
- Sepal Width: {sepal_width} cm
- Petal Length: {petal_length} cm
- Petal Width: {petal_width} cm

Predicted Species: {species.capitalize()}
Confidence: {confidence:.2f}%

Class Probabilities:
"""

for i, prob in enumerate(probabilities[0]):
    report_text += f"- {iris.target_names[i].capitalize()}: {prob * 100:.2f}%\n"

# Download button
st.download_button("📥 Download Report", report_text, file_name="iris_prediction.txt")
