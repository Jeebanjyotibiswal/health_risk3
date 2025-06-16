from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import os
from reportlab.pdfgen import canvas
from io import BytesIO

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define symptoms list
symptoms = [
    "Chest Pain", "Shortness of Breath", "Irregular Heartbeat", "Fatigue & Weakness",
    "Dizziness", "Swelling (Edema)", "Pain in Neck/Jaw/Shoulder/Back", "Excessive Sweating",
    "Persistent Cough", "Nausea/Vomiting", "High Blood Pressure", "Chest Discomfort (Activity)",
    "Cold Hands/Feet", "Snoring/Sleep Apnea", "Anxiety/Feeling of Doom"
]

# Home page route
@app.route("/")
def index():
    return render_template("index.html", symptoms=symptoms, page_title="Health Risk Predictor | Home")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []
        symptom_values = {}

        # Collect symptom values
        for symptom in symptoms:
            value = int(request.form.get(symptom, 0))
            input_data.append(value)
            symptom_values[symptom] = value

        # Get age input
        age = int(request.form.get("Age", 30))
        input_data.append(age)

        final_input = np.array(input_data).reshape(1, -1)

        # Predict risk
        raw_prediction = model.predict(final_input)[0]
        prediction = max(0, min(100, round(raw_prediction, 2)))

        # Top contributing factors
        top_factors = []
        if age > 45:
            top_factors.append(f"Age ({age}) increased risk by {min(20, age//5)}%")

        symptom_contributors = [s for s in symptoms if symptom_values.get(s, 0) == 1][:3]
        for i, symptom in enumerate(symptom_contributors):
            top_factors.append(f"{symptom} contributed {15 - i*5}%")

        # Risk level and recommendation
        if prediction > 70:
            risk_level = "high"
            recommendation = "Immediate consultation recommended"
        elif prediction > 30:
            risk_level = "moderate"
            recommendation = "Preventive measures advised"
        else:
            risk_level = "low"
            recommendation = "Maintain healthy lifestyle"

        return render_template("result.html",
                               prediction=prediction,
                               risk_level=risk_level,
                               recommendation=recommendation,
                               top_factors=top_factors,
                               age=age,
                               page_title=f"Results | {prediction}% Risk")

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template("index.html",
                               symptoms=symptoms,
                               prediction="Error",
                               error_message="Could not process your request. Please try again.")

# Report PDF download route
@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        # Get values from form
        prediction = request.form.get("prediction")
        risk_level = request.form.get("risk_level")
        recommendation = request.form.get("recommendation")
        age = request.form.get("age")
        top_factors = request.form.getlist("top_factors")

        # Create PDF in memory
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(180, 800, "Health Risk Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"Prediction Score: {prediction}%")
        c.drawString(50, 750, f"Risk Level: {risk_level}")
        c.drawString(50, 730, f"Recommendation: {recommendation}")
        c.drawString(50, 710, f"Age: {age}")

        c.drawString(50, 690, "Top Risk Factors:")
        y = 670
        for factor in top_factors:
            c.drawString(70, y, f"- {factor}")
            y -= 20

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        return send_file(pdf_buffer, as_attachment=True, download_name="Health_Risk_Report.pdf", mimetype='application/pdf')

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return "Could not generate report. Please try again.", 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)
