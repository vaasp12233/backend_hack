import streamlit as st
import pickle
import re
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Fraud Shield Pro", page_icon="🛡️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .safe-box { border: 2px solid #28a745; padding: 15px; border-radius: 10px; background-color: #f0fff4; }
    .suspicious-box { border: 2px solid #ffc107; padding: 15px; border-radius: 10px; background-color: #fff9e6; }
    .highrisk-box { border: 2px solid #dc3545; padding: 15px; border-radius: 10px; background-color: #fff5f5; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    try:
        models['model_risk'] = pickle.load(open('model_risk.pkl', 'rb'))
        models['tfidf'] = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        models['label_encoder'] = pickle.load(open('label_encoder.pkl', 'rb'))
    except FileNotFoundError as e:
        st.error(f"Missing model file: {e}. Please ensure model_risk.pkl, tfidf_vectorizer.pkl, and label_encoder.pkl are in the app directory.")
        return None

    # Try to load multi-class model, but it's optional (fallback used if missing)
    try:
        models['model_type'] = pickle.load(open('model_type.pkl', 'rb'))
        models['type_model_available'] = True
    except FileNotFoundError:
        models['type_model_available'] = False
        st.warning("Fraud type model not found. Using rule‑based fallback for fraud type detection.")
    return models

models = load_models()

# --- TEXT CLEANING (must match training) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'\b\d{10}\b', ' PHONE ', text)
    text = re.sub(r'rs\.?\s*\d+|\d+\s*(lakh|crore|thousand)|₹\s*\d+', ' AMOUNT ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- KEYWORD HIGHLIGHTING ---
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

# --- FALLBACK RULE‑BASED FRAUD TYPE CLASSIFIER ---
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

# --- PREVENTIVE TIPS ---
def get_preventive_tips(fraud_type):
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
    return tips.get(fraud_type, tips["Others"])

# --- MAIN UI ---
st.title("🛡️ AI Fraud Shield Pro")
st.markdown("### Real-time SMS/Message Fraud Risk Analysis")

user_input = st.text_area("📨 Paste the message you received:", height=150)

if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    if not user_input or len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter a valid message (at least 10 characters)")
    elif models is None:
        st.error("Models not loaded. Please check model files.")
    else:
        with st.spinner("Analyzing..."):
            # Clean and vectorize
            cleaned = clean_text(user_input)
            vec = models['tfidf'].transform([cleaned])

            # Scam probability (binary classifier)
            prob_spam = models['model_risk'].predict_proba(vec)[0][1] * 100

            # Risk classification
            if prob_spam < 30:
                risk = "Safe"
            elif prob_spam < 75:
                risk = "Suspicious"
            else:
                risk = "High Risk"

            # Fraud type detection
            if risk == "Safe":
                fraud_type = "None"
            else:
                if models.get('type_model_available'):
                    # Use trained multi-class model
                    type_encoded = models['model_type'].predict(vec)[0]
                    detailed_type = models['label_encoder'].inverse_transform([type_encoded])[0]
                    # Map detailed type to one of the five categories if needed
                    # (Assuming the model already outputs one of the five; if not, add mapping)
                    fraud_type = detailed_type  # adjust if necessary
                else:
                    # Fallback rule-based
                    fraud_type = rule_based_fraud_type(user_input)

            # Keyword highlighting
            highlighted = highlight_keywords(user_input)

            # Preventive tips
            tips = get_preventive_tips(fraud_type if risk != "Safe" else "Others")

            # Display results
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Scam Probability", f"{prob_spam:.1f}%")
            col2.metric("Risk Level", risk)
            col3.metric("Fraud Type", fraud_type)

            st.progress(int(prob_spam))

            if risk == "Safe":
                st.markdown(f"""
                <div class="safe-box">
                    <h4 style='color:#28a745;'>✅ Safe Message</h4>
                    <p>This message appears legitimate. However, always stay vigilant!</p>
                </div>
                """, unsafe_allow_html=True)
            elif risk == "Suspicious":
                st.markdown(f"""
                <div class="suspicious-box">
                    <h4 style='color:#856404;'>⚠️ Suspicious Message</h4>
                    <p>This message shows suspicious patterns. Do not engage with the sender.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="highrisk-box">
                    <h4 style='color:#dc3545;'>🚨 HIGH RISK - SCAM DETECTED</h4>
                    <p>This is likely a <b>{fraud_type}</b>. DO NOT respond or click any links.</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("🔍 Pattern Analysis")
            st.markdown(f"**Highlighted Text:** {highlighted}", unsafe_allow_html=True)

            st.subheader("💡 Preventive Guidance")
            for tip in tips:
                st.write(tip)

            st.info("📞 Official Helpline: **1930** | [Report Online](https://www.cybercrime.gov.in)")

# --- FOOTER ---
st.divider()
st.caption("🛡️ AI Fraud Shield Pro | Powered by Machine Learning | Report Suspicious Messages to 1930")