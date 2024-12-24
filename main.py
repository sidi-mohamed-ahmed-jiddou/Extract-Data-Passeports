import cv2
import requests
from paddleocr import PaddleOCR
from transformers import pipeline
import json
import re
import numpy as np
import streamlit as st

# Function to perform OCR using PaddleOCR
def ocr_with_paddle(img):
    finaltext = ''
    ocr = PaddleOCR(lang='en', use_angle_cls=True)
    result = ocr.ocr(img)

    for i in range(len(result[0])):
        text = result[0][i][1][0]
        finaltext += ' ' + text
    return finaltext

# Function to generate OCR from the image
def generate_ocr(img):
    text_output = ''
    if img is not None:
        text_output = ocr_with_paddle(img)
        return text_output
    else:
        st.error("Please upload an image!!!!")
        return None


# Dictionnaire pour convertir les noms des mois en numéros
MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

# Fonction pour corriger les erreurs communes dans les noms des mois
def correct_month_errors(date_str):
    corrections = {
        " AUG/A0UT": "AUG",  # Correction pour les erreurs spécifiques
        "JAN/JAN": "JAN",   # Suppression des doublons
    }
    for error, correction in corrections.items():
        date_str = date_str.replace(error, correction)
    return date_str

# Fonction pour extraire et convertir la date
def extract_date(date_str):
    
    # Corriger les erreurs dans les noms des mois
    date_str = correct_month_errors(date_str)
    # Supprimer les doublons (ex: "JANJAN" devient "JAN")
    date_str = re.sub(r'([A-Za-z]{3})/\1', r'\1', date_str)
    # Si la date contient le mois et l'année sans séparateur
    if re.match(r'\d{2}[A-Za-z]{3}\d{4}', date_str):
        # Extraire les parties de la date (jour, mois, année)
        match = re.search(r'(\d{2})([A-Za-z]{3})(\d{4})', date_str)
        if match:
            day = match.group(1)
            month = MONTHS.get(match.group(2).upper(), "00")
            year = match.group(3)
            return f"{day}/{month}/{year}"
        
    date_str = date_str.replace(" ", "")
    return date_str

# Fonction pour extraire des informations structurées avec le modèle de question-réponse
def extract_information_with_model(ocr_text):
    pipe = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

    questions = {
    "name": "What is the surname/Nom of the person?",
    "given names": "What are the given names/Prénoms of the person?",
    "passport_number": "What is the passport number?",
    "date_of_birth": "What is the date of birth of the person?",
    "date_of_issue": "What is the date of issue of the passport?",
    "date_of_expiration": "What is the expiration date of the passport?",
    "nationality": "What is the nationality/nationalité of the person?",
    "place_of_birth": "What is the place of birth/Lieu de naissance of the person?",
    "authority_of_issuance": "What is the authority/autorité of issuance for the passport?",
}

    extracted_data = {}
    for field, question in questions.items():
        answer = pipe(question=question, context=ocr_text)
        extracted_data[field] = answer['answer']

    # Nettoyer les dates

    extracted_data["date_of_birth"] = extract_date(extracted_data["date_of_birth"])
    extracted_data["date_of_issue"] = extract_date(extracted_data["date_of_issue"])
    extracted_data["date_of_expiration"] = extract_date(extracted_data["date_of_expiration"])

    return json.dumps(extracted_data, indent=4)

# Streamlit UI and main function
def main():
    st.title("Passport Information Extractor")

    uploaded_file = st.file_uploader("Upload an image of a passport", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Show the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Process the image
        output_text = generate_ocr(img)
        if output_text:

            # Extract structured information using the QA model
            output_data = extract_information_with_model(output_text)
            
            # Display the extracted data in a JSON format
            st.subheader("Extracted Information:")
            st.json(json.loads(output_data))
        else:
            st.warning("OCR did not extract any text.")
    else:
        st.info("Please upload an image to start the process.")

if __name__ == '__main__':
    main()
