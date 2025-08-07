# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# # Load env variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize Flask
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


# # Load model
# model, scaler = joblib.load('heart_model.pkl')

# # LangChain LLM
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

# def generate_explanation(risk, recommendations):
#     prompt = f"""
#     A patient has a heart disease risk classified as {risk}.
#     Recommendations: {', '.join(recommendations)}.
#     Explain this in simple, friendly, and motivating language for the patient.
#     """
#     response = llm.invoke(prompt)
#     return response.content

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json['features']  # e.g. [63,1,3,145,233,...]
#     data_scaled = scaler.transform([data])
#     prediction = model.predict(data_scaled)[0]
#     risk = "High Risk" if prediction == 1 else "Low Risk"

#     # Recommendations
#     recommendations = [
#         "Maintain a healthy diet",
#         "Exercise regularly",
#         "Get routine health checkups"
#     ]
#     if prediction == 1:
#         recommendations = [
#             "Consult a cardiologist immediately",
#             "Maintain a low-sodium diet",
#             "Increase physical activity under medical supervision",
#             "Monitor blood pressure and cholesterol regularly"
#         ]

#     # AI explanation
#     explanation = generate_explanation(risk, recommendations)

#     return jsonify({
#         "prediction": int(prediction),
#         "risk": risk,
#         "recommendations": recommendations,
#         "ai_explanation": explanation
#     })

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import google.generativeai as genai

# ================== CONFIG ===================
# Load model and scaler
model, scaler = joblib.load("heart_model.pkl")

# Configure Gemini API
genai.configure(api_key="OPENAI_API_KEY")  # Replace with your key

app = Flask(__name__)
CORS(app)  # Enable CORS for React

# Function to generate explanation using Gemini
def generate_explanation(risk, recommendations):
    prompt = f"""
    A patient is assessed for heart disease risk.
    Risk Level: {risk}
    Recommendations: {', '.join(recommendations) if recommendations else 'None'}
    Explain this result in simple terms for the patient.
    """
    
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)
        return response.text if response else "No explanation generated."
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

# ================== API ROUTES ===================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get("features")
        if not data:
            return jsonify({"error": "No features provided"}), 400

        # Scale input
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        risk = "High Risk" if prediction == 1 else "Low Risk"

        # Basic recommendations
        recommendations = []
        if risk == "High Risk":
            recommendations = [
                "Consult a cardiologist",
                "Adopt a heart-healthy diet",
                "Increase physical activity"
            ]
        else:
            recommendations = [
                "Maintain your healthy lifestyle",
                "Continue regular checkups"
            ]

        # Generate explanation from Gemini
        explanation = generate_explanation(risk, recommendations)

        return jsonify({
            "risk": risk,
            "recommendations": recommendations,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
