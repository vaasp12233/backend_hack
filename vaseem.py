import os
import pickle
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

# ------------------------------------------------------------
# Load trained models (if files exist)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model_risk = load_pickle('model_risk.pkl')
model_type = load_pickle('model_type.pkl')
tfidf = load_pickle('tfidf_vectorizer.pkl')
le = load_pickle('label_encoder.pkl')

if model_risk is None or tfidf is None or le is None:
    print("WARNING: Some essential model files are missing. Please check.")

# ------------------------------------------------------------
# Text preprocessing (must match training)
# ------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'\b\d{10}\b', ' PHONE ', text)
    text = re.sub(r'rs\.?\s*\d+|\d+\s*(lakh|crore|thousand)|₹\s*\d+', ' AMOUNT ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------------------------------
# Map detailed fraud type to one of the five required categories
# ------------------------------------------------------------
def map_to_five_categories(detailed_type):
    dt = detailed_type.lower()
    if any(k in dt for k in ['upi', 'bank', 'payment', 'atm', 'debit', 'credit', 'account']):
        return 'UPI Fraud'
    elif any(k in dt for k in ['job', 'work', 'employ', 'career', 'salary', 'income']):
        return 'Job Scam'
    elif any(k in dt for k in ['lottery', 'prize', 'won', 'winner', 'kbc', 'draw', 'lucky']):
        return 'Lottery Scam'
    elif any(k in dt for k in ['phish', 'kyc', 'verify', 'update', 'aadhaar', 'pan', 'link',
                                'courier', 'parcel', 'dhl', 'fedex', 'netflix', 'prime',
                                'subscription', 'ott', 'education', 'govt', 'matrimony']):
        return 'Phishing'
    else:
        return 'Others'

# ------------------------------------------------------------
# Fallback rule‑based classifier (used if model_type is missing)
# ------------------------------------------------------------
def rule_based_fraud_type(message):
    msg = message.lower()
    if any(k in msg for k in ['upi', 'gpay', 'phonepe', 'pin', 'otp', 'bank', 'account', 'atm']):
        return 'UPI Fraud'
    elif any(k in msg for k in ['job', 'work from home', 'salary', 'part time', 'earning']):
        return 'Job Scam'
    elif any(k in msg for k in ['lottery', 'won', 'prize', 'kbc', 'winner']):
        return 'Lottery Scam'
    elif any(k in msg for k in ['kyc', 'update', 'verify', 'aadhaar', 'pan', 'link', 'click']):
        return 'Phishing'
    else:
        return 'Others'

# ------------------------------------------------------------
# Keyword highlighting (returns HTML)
# ------------------------------------------------------------
def highlight_keywords(text):
    keywords = {
        'upi': 'red', 'gpay': 'red', 'phonepe': 'red', 'pin': 'red', 'otp': 'red',
        'lottery': 'orange', 'won': 'orange', 'prize': 'orange', 'kbc': 'orange',
        'job': 'blue', 'salary': 'blue', 'work from home': 'blue',
        'kyc': 'purple', 'update': 'purple', 'verify': 'purple', 'link': 'purple',
        'bank': 'green', 'account': 'green', 'blocked': 'red', 'suspended': 'red',
        'courier': 'brown', 'parcel': 'brown', 'dhl': 'brown', 'fedex': 'brown',
        'netflix': 'violet', 'amazon prime': 'violet', 'subscription': 'violet',
        'matrimony': 'pink', 'shaadi': 'pink', 'caste': 'pink',
        'education': 'teal', 'scholarship': 'teal', 'exam': 'teal'
    }
    highlighted = text
    for word, color in keywords.items():
        pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
        highlighted = pattern.sub(f'<span style="color:{color};font-weight:bold;">\\1</span>', highlighted)
    return highlighted

# ------------------------------------------------------------
# Preventive tips based on the broad category
# ------------------------------------------------------------
def get_preventive_tips(category):
    tips = {
        "UPI Fraud": [
            "❌ NEVER share UPI PIN or OTP with anyone",
            "✅ Always check VPA/UPI ID before making payment",
            "⚠️ Banks NEVER ask for KYC via SMS links",
            "📞 Report suspicious UPI transactions to 1930 immediately"
        ],
        "Job Scam": [
            "❌ No genuine job asks for registration fees",
            "✅ Verify company on LinkedIn before applying",
            "⚠️ Be cautious of 'work from home' jobs promising high income",
            "📞 Report job scams on cybercrime.gov.in"
        ],
        "Lottery Scam": [
            "❌ You cannot win a lottery you never entered",
            "✅ KBC/Amazon never inform winners via SMS",
            "⚠️ Never pay 'processing fees' to claim prizes",
            "📞 Report lottery scams to 1930"
        ],
        "Phishing": [
            "❌ Banks never ask for passwords or OTP via SMS",
            "✅ Always type bank URLs manually, don't click links",
            "⚠️ Check for spelling mistakes in URLs",
            "📞 Report phishing to report.phishing@gmail.com"
        ],
        "Others": [
            "❌ Don't click on unknown links",
            "✅ Verify sender identity before responding",
            "⚠️ Never share personal information via SMS",
            "📞 Report suspicious messages to 1930"
        ]
    }
    return tips.get(category, tips["Others"])

# ------------------------------------------------------------
# API endpoint
# ------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = data['message']

    # Clean and vectorize
    cleaned = clean_text(message)
    vec = tfidf.transform([cleaned])

    # Scam probability (binary classifier)
    prob_spam = model_risk.predict_proba(vec)[0][1] * 100

    # Risk classification
    if prob_spam < 30:
        risk = "Safe"
    elif prob_spam < 75:
        risk = "Suspicious"
    else:
        risk = "High Risk"

    # Fraud type detection
    if risk == "Safe":
        detailed_type = "None"
        broad_category = "None"
    else:
        if model_type is not None:
            # Use trained multi-class model
            type_encoded = model_type.predict(vec)[0]
            detailed_type = le.inverse_transform([type_encoded])[0]
            broad_category = map_to_five_categories(detailed_type)
        else:
            # Fallback rule-based
            broad_category = rule_based_fraud_type(message)
            detailed_type = broad_category  # or could be more detailed

    # Highlighted text
    highlighted = highlight_keywords(message)

    # Preventive tips
    tips = get_preventive_tips(broad_category if risk != "Safe" else "Others")

    response = {
        'scam_probability': round(prob_spam, 2),
        'risk_classification': risk,
        'fraud_type': broad_category,          # one of the five categories
        'detailed_fraud_type': detailed_type,  # original model output (if available)
        'highlighted_text': highlighted,
        'preventive_tips': tips,
        'helpline': '1930',
        'cybercrime_website': 'https://www.cybercrime.gov.in'
    }
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
