from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import os
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from datetime import datetime

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

        # Collect personal details
        name = request.form.get("Name", "Anonymous")
        age = int(request.form.get("Age", 30))
        blood_group = request.form.get("BloodGroup", "Not specified")

        # Collect symptom values
        for symptom in symptoms:
            value = int(request.form.get(symptom, 0))
            input_data.append(value)
            symptom_values[symptom] = value

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
            risk_level = "High"
            recommendation = "Immediate consultation with a cardiologist recommended"
        elif prediction > 30:
            risk_level = "Moderate"
            recommendation = "Preventive measures and regular checkups advised"
        else:
            risk_level = "Low"
            recommendation = "Maintain healthy lifestyle with annual checkups"

        return render_template("result.html",
                            prediction=prediction,
                            risk_level=risk_level,
                            recommendation=recommendation,
                            top_factors=top_factors,
                            name=name,
                            age=age,
                            blood_group=blood_group,
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
        name = request.form.get("name")
        age = request.form.get("age")
        blood_group = request.form.get("blood_group")
        top_factors = request.form.getlist("top_factors")
        current_date = datetime.now().strftime("%d %B, %Y")
        current_time = datetime.now().strftime("%I:%M %p")

        # Create PDF in memory
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor("#1a5276"),
            spaceAfter=20,
            alignment=1  # Center aligned
        )
        
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor("#2874a6"),
            spaceAfter=10,
            spaceBefore=20,
            underline=1
        )
        
        # Header with improved visibility
        title = Paragraph("HEALTH RISK ASSESSMENT REPORT", title_style)
        title.wrapOn(c, 500, 50)
        title.drawOn(c, 50, 750)
        
        # Date and time - properly positioned
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawRightString(550, 780, current_date)
        c.drawRightString(550, 765, current_time)
        
        # Patient Information section with improved heading
        section = Paragraph("PATIENT INFORMATION", section_style)
        section.wrapOn(c, 500, 50)
        section.drawOn(c, 50, 700)
        
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        c.drawString(50, 680, f"Name: {name}")
        c.drawString(50, 660, f"Age: {age}")
        c.drawString(50, 640, f"Blood Group: {blood_group}")
        
        # Results section with improved heading
        section = Paragraph("ASSESSMENT RESULTS", section_style)
        section.wrapOn(c, 500, 50)
        section.drawOn(c, 50, 600)
        
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        c.drawString(50, 580, f"Risk Score: {prediction}%")
        
        # Color code risk level
        if float(prediction) > 70:
            risk_color = colors.HexColor("#e74c3c")  # Red
        elif float(prediction) > 30:
            risk_color = colors.HexColor("#f39c12")  # Orange
        else:
            risk_color = colors.HexColor("#27ae60")  # Green
            
        c.drawString(50, 560, "Risk Level: ")
        c.setFillColor(risk_color)
        c.drawString(120, 560, risk_level)
        c.setFillColor(colors.black)
        c.drawString(50, 540, f"Recommendation: {recommendation}")
        
        # Risk Factors section with improved heading
        section = Paragraph("KEY RISK FACTORS", section_style)
        section.wrapOn(c, 500, 50)
        section.drawOn(c, 50, 500)
        
        y = 480
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        for factor in top_factors:
            c.drawString(70, y, f"• {factor}")
            y -= 20
        
        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 100, "This report is generated by Health Risk Predictor")
        c.drawString(50, 85, "Note: This is not a medical diagnosis. Please consult your doctor.")
        
        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        return send_file(pdf_buffer, as_attachment=True, 
                        download_name=f"Health_Report_{name.replace(' ', '_')}.pdf", 
                        mimetype='application/pdf')

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return "Could not generate report. Please try again.", 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)
