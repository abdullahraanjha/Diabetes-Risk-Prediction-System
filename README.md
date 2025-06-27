# Diabetes-Risk-Prediction-System
**html** 
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Diabetes Predictor</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <div class="login">
    <h1>Diabetes Risk Prediction</h1>
    <form action="{{ url_for('predict') }}" method="post">
      <input type="text" name="Pregnancies" placeholder="Pregnancies" required>
      <input type="text" name="Glucose" placeholder="Glucose" required>
      <input type="text" name="BloodPressure" placeholder="Blood Pressure" required>
      <input type="text" name="SkinThickness" placeholder="Skin Thickness" required>
      <input type="text" name="Insulin" placeholder="Insulin" required>
      <input type="text" name="BMI" placeholder="BMI" required>
      <input type="text" name="DiabetesPedigreeFunction" placeholder="Diabetes Pedigree Function" required>
      <input type="text" name="Age" placeholder="Age" required>
      <button type="submit" class="btn">Predict</button>
    </form>
    <h1 id="predi">{{ prediction_text }}</h1>
  </div>
</body>
</html>

**Css**
body {
  background: #f0f0f0;
  font-family: Arial, sans-serif;
}
.login {
  width: 400px;
  margin: 100px auto;
  padding: 30px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
h1 {
  text-align: center;
  font-weight: bold;
}
input, button.btn {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
}
button.btn {
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
}
#predi {
  text-align: center;
  font-weight: bold;
  margin-top: 20px;
}

**Model**

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("diabetes.csv")  # Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Features and labels
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Optional: check accuracy
accuracy = model.score(x_test, y_test)
print("Model Accuracy:", accuracy)

**App**
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
    return render_template("index.html", prediction_text="The Prediction is: {}".format(result))

if __name__ == "__main__":
    app.run(debug=True)
