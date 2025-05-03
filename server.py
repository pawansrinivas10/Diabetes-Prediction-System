from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow all origins

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
        features = np.array([[
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]])
        prediction = model.predict(features)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
