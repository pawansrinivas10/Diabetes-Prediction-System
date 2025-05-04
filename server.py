from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded.")

@app.route("/")
def home():
    return "Diabetes Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[data["Pregnancies"],
                              data["Glucose"],
                              data["BloodPressure"],
                              data["SkinThickness"],
                              data["Insulin"],
                              data["BMI"],
                              data["DiabetesPedigreeFunction"],
                              data["Age"]]])
        prediction = model.predict(features)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host='0.0.0.0', port=port)
