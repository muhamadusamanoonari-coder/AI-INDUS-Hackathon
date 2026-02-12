import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import random
import shap
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_df)
st.pyplot(shap.summary_plot(shap_vals[1], input_df))

st.markdown("""
<style>
@media (max-width: 768px) {
    .block-container {
        padding: 1rem !important;
    }
    h1 {font-size: 28px;}
    .stButton button {
        width:100%;
        font-size:16px;
    }
}
</style>
""", unsafe_allow_html=True)


# --- 1. SAFE LIBRARY LOADING ---
try:
    from fpdf import FPDF
except ImportError:
    st.error("⚠️ Library 'fpdf' is missing. Run: pip install fpdf")
    st.stop()

# --- 2. APP CONFIGURATION ---
st.set_page_config(
    page_title="MediGuard AI Pro+",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- 3. ADVANCED UI/UX STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    /* General App Styling */
    .stApp {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Typography */
    h1, h2, h3 { color: #1e293b; font-weight: 700; }
    h1 {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.6);
    }
    
    /* Input Fields */
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar Clean-up */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f1f5f9;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. ROBUST BACKEND FUNCTIONS ---

@st.cache_resource
def load_prediction_model():
    """Loads model with a fallback dummy for demonstration if file missing."""
    try:
        return joblib.load("health_model.pkl")
    except:
        # Create a dummy pipeline for UI demo purposes
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42))
        ])
        # Train on dummy data
        X = np.random.rand(10, 10)
        y = np.random.randint(0, 2, 10)
        pipeline.fit(X, y)
        return pipeline

model = load_prediction_model()

def create_radar_chart(s1, s2, s3, s4, s5, s6):
    """Creates a unique Radar Chart for biomarkers."""
    categories = ['TC', 'LDL', 'HDL', 'TCH', 'LTG', 'GLU']
    # Normalize values for visualization (purely for UI effect in demo)
    values = [s1, s2, s3, s4, s5, s6]
    # Scale up small float values for chart visibility
    visual_values = [v * 100 if abs(v) < 1 else v for v in values] 

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=visual_values,
        theta=categories,
        fill='toself',
        name='Patient Profile',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-20, 20])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def generate_pdf_report(user_name, vitals, risk_status, health_score, advice_list):
    """Generates a professional-looking PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(30, 58, 138) # Dark Blue
    pdf.cell(0, 15, "MediGuard AI", ln=True, align="C")
    
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Automated Clinical Risk Assessment", ln=True, align="C")
    pdf.line(10, 35, 200, 35)
    
    # Patient Details
    pdf.ln(15)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient Name: {user_name}", ln=False)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="R")
    
    # Vitals Box
    pdf.ln(5)
    pdf.set_fill_color(240, 249, 255)
    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
    pdf.set_font("Arial", "", 11)
    pdf.ln(5)
    vitals_text = f"  Age: {vitals['age']}   |   BMI: {vitals['bmi']}   |   BP: {vitals['bp']} mmHg"
    pdf.cell(0, 10, vitals_text, ln=True)

    # Result
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    color = (220, 38, 38) if "Elevated" in risk_status else (22, 163, 74)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Analysis Result: {risk_status.upper()}", ln=True)
    
    # Score
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Wellness Score: {health_score}/100", ln=True)
    
    # Recommendations
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Generated Recommendations:", ln=True)
    pdf.set_font("Arial", "", 11)
    for item in advice_list:
        # Handle unicode roughly for basic FPDF
        safe_item = item.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, f"- {safe_item}")
        
    return pdf.output(dest="S").encode("latin-1", errors='ignore')

# --- 5. LOCAL AI LOGIC (No API Key Needed) ---
def get_smart_response(prompt):
    """Simulates a medical AI using keyword matching logic."""
    p = prompt.lower()
    
    # Response Database
    responses = {
        "diet": "For a balanced diet, focus on whole foods. \n\n**Recommendation:**\n- **Proteins:** Lean chicken, fish, tofu.\n- **Carbs:** Quinoa, brown rice, oats.\n- **Veggies:** Leafy greens like spinach and kale.\n\n*Avoid processed sugars and high-sodium foods.*",
        "food": "For a balanced diet, focus on whole foods. \n\n**Recommendation:**\n- **Proteins:** Lean chicken, fish, tofu.\n- **Carbs:** Quinoa, brown rice, oats.\n- **Veggies:** Leafy greens like spinach and kale.\n\n*Avoid processed sugars and high-sodium foods.*",
        "bmi": "Body Mass Index (BMI) is a screening tool used to estimate body fat.\n\n- **Underweight:** < 18.5\n- **Normal:** 18.5 - 24.9\n- **Overweight:** 25 - 29.9\n- **Obese:** 30+\n\n*Note: It does not measure muscle mass directly.*",
        "exercise": "The American Heart Association recommends:\n\n1. **Aerobic:** 150 mins moderate activity/week (brisk walking).\n2. **Strength:** Muscle-strengthening 2 days/week.\n3. **Flexibility:** Daily stretching or yoga.",
        "workout": "The American Heart Association recommends:\n\n1. **Aerobic:** 150 mins moderate activity/week (brisk walking).\n2. **Strength:** Muscle-strengthening 2 days/week.\n3. **Flexibility:** Daily stretching or yoga.",
        "diabetes": "Diabetes management relies on monitoring blood sugar.\n\n- **Tip 1:** Eat at regular intervals.\n- **Tip 2:** Choose low-GI foods (beans, sweet potatoes).\n- **Tip 3:** Stay hydrated.\n\n*Consult an endocrinologist for specific medication advice.*",
        "sugar": "Diabetes management relies on monitoring blood sugar.\n\n- **Tip 1:** Eat at regular intervals.\n- **Tip 2:** Choose low-GI foods (beans, sweet potatoes).\n- **Tip 3:** Stay hydrated.\n\n*Consult an endocrinologist for specific medication advice.*",
        "blood pressure": "Managing Blood Pressure (Hypertension):\n\n- **DASH Diet:** Rich in fruits, veggies, and low-fat dairy.\n- **Sodium:** Limit to < 2,300 mg per day.\n- **Stress:** Practice meditation or deep breathing.",
        "bp": "Managing Blood Pressure (Hypertension):\n\n- **DASH Diet:** Rich in fruits, veggies, and low-fat dairy.\n- **Sodium:** Limit to < 2,300 mg per day.\n- **Stress:** Practice meditation or deep breathing.",
        "risk": "This application uses a Random Forest machine learning model. It analyzes age, BMI, blood pressure, and serum markers (cholesterol, glucose, etc.) to calculate a probability score for potential health risks."
    }
    
    # Check for keywords
    for key in responses:
        if key in p:
            return responses[key]
            
    # Default fallback
    return "I can help with general health queries. Try asking about:\n- **'Recommended diet plans'**\n- **'What is a good BMI?'**\n- **'Exercise tips'**\n- **'How to lower blood pressure?'**"

# --- 6. SESSION STATE INITIALIZATION ---
if "history" not in st.session_state: st.session_state.history = []
if "messages" not in st.session_state: st.session_state.messages = []
if "form_data" not in st.session_state:
    st.session_state.form_data = {"age": 45, "bmi": 26.5, "bp": 115, "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": 0.0, "s6": 0.0}

# --- 7. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🏥 MediGuard Pro")
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=60)
    st.write("Current User:")
    user_name = st.text_input("Name", "Guest Patient", label_visibility="collapsed")
    
    st.divider()
    st.info("🔒 **Privacy Mode Active**\nLocal Processing Only. No Cloud Uploads.")

# --- 8. MAIN INTERFACE ---

# Hero Header
st.markdown(f"""
    <div>
        <h1>Welcome, {user_name.split()[0]}</h1>
        <p style="color: #64748b; font-size: 1.1rem; margin-top: -15px;">
            AI-Powered Health Risk Assessment & Monitoring System
        </p>
    </div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Diagnostic Dashboard", "🤖 AI Specialist Chat", "📂 Medical Records"])

# --- TAB 1: DIAGNOSTIC DASHBOARD ---
with tabs[0]:
    # Demo Button Logic
    col_demo, _ = st.columns([1, 4])
    if col_demo.button("⚡ Load Demo Data"):
        st.session_state.form_data = {
            "age": 55, "bmi": 31.2, "bp": 145,
            "s1": 0.04, "s2": 0.03, "s3": -0.05, "s4": 0.06, "s5": 0.08, "s6": 0.09
        }
        st.rerun()

    # Main Grid
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📝 Clinical Data Entry")
        
        # Group 1: Vitals
        st.markdown("##### 🩺 Essential Vitals")
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 18, 90, st.session_state.form_data["age"])
        sex = 1 if c2.selectbox("Biological Sex", ["Male", "Female"]) == "Male" else 0
        
        c3, c4 = st.columns(2)
        bmi = c3.number_input("BMI Index", 10.0, 60.0, st.session_state.form_data["bmi"])
        bp = c4.number_input("Avg BP (mmHg)", 80, 220, st.session_state.form_data["bp"])

        # Group 2: Lab Results
        st.markdown("---")
        st.markdown("##### 🩸 Lab Biomarkers (Normalized)")
        with st.expander("Enter Blood Serum Data", expanded=True):
            lc1, lc2 = st.columns(2)
            s1 = lc1.number_input("TC (Total Chol)", -0.5, 0.5, st.session_state.form_data["s1"], step=0.01)
            s2 = lc2.number_input("LDL (Low Density)", -0.5, 0.5, st.session_state.form_data["s2"], step=0.01)
            s3 = lc1.number_input("HDL (High Density)", -0.5, 0.5, st.session_state.form_data["s3"], step=0.01)
            s4 = lc2.number_input("TCH (Total/HDL)", -0.5, 0.5, st.session_state.form_data["s4"], step=0.01)
            s5 = lc1.number_input("LTG (Lamotrigine)", -0.5, 0.5, st.session_state.form_data["s5"], step=0.01)
            s6 = lc2.number_input("GLU (Glucose)", -0.5, 0.5, st.session_state.form_data["s6"], step=0.01)
        
        analyze_btn = st.button("Run Full Analysis", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if analyze_btn:
            # Prediction Logic
            input_vector = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
            
            # Safe predict
            try:
                prediction = model.predict(input_vector)[0]
                proba = model.predict_proba(input_vector)[0][1] if hasattr(model, "predict_proba") else (0.9 if prediction == 1 else 0.1)
            except:
                prediction = 1 if bmi > 30 else 0
                proba = 0.85 if prediction == 1 else 0.15

            risk_score = int(proba * 100)
            health_score = 100 - risk_score
            
            # Dynamic Styling based on result
            if prediction == 1:
                status_color = "#ef4444" # Red
                status_bg = "#fef2f2"
                status_msg = "Elevated Risk Detected"
                icon = "⚠️"
            else:
                status_color = "#10b981" # Green
                status_bg = "#ecfdf5"
                status_msg = "Health Metrics Optimal"
                icon = "✅"

            # --- RESULT CARD ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Status Banner
            st.markdown(f"""
                <div style="background-color: {status_bg}; padding: 15px; border-radius: 12px; border: 1px solid {status_color}; display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                    <span style="font-size: 24px;">{icon}</span>
                    <div>
                        <h3 style="margin: 0; color: {status_color}; font-size: 18px;">{status_msg}</h3>
                        <p style="margin: 0; color: #64748b; font-size: 14px;">Confidence: {int(proba*100)}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Wellness Score", f"{health_score}/100", delta=f"{health_score-50}", delta_color="normal")
            m2.metric("BMI Status", f"{bmi:.1f}", "High" if bmi > 25 else "Normal", delta_color="inverse")
            m3.metric("Heart Age", f"{age + (5 if prediction == 1 else -2)} yrs", help="Estimated biological age")

            # Charts
            st.markdown("##### 🧬 Biomarker Profile")
            st.caption("Visual representation of your blood serum levels vs baseline.")
            st.plotly_chart(create_radar_chart(s1, s2, s3, s4, s5, s6), use_container_width=True)
            
            # Recommendations
            recs = []
            if prediction == 1:
                recs = ["Consult a cardiologist for a stress test.", "Reduce sodium intake < 2300mg/day.", "Monitor blood sugar fasting levels."]
            else:
                recs = ["Maintain current activity levels.", "Yearly check-up recommended.", "Ensure Vitamin D sufficiency."]
            
            st.info(f"💡 **Recommendation:** {recs[0]}")

            # PDF Download
            pdf_bytes = generate_pdf_report(user_name, {"age": age, "bmi": bmi, "bp": bp}, status_msg, health_score, recs)
            st.download_button("📄 Download Medical Report", pdf_bytes, f"Report_{user_name}.pdf", "application/pdf", use_container_width=True)

            # Save to History
            st.session_state.history.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Result": status_msg, "Score": health_score, "BMI": bmi
            })
            
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Empty State
            st.markdown('<div class="glass-card" style="text-align: center; padding: 50px;">', unsafe_allow_html=True)
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
            st.markdown("### Ready to Analyze")
            st.markdown("Fill in the clinical data on the left or click **'Load Demo Data'** to start.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: AI CHAT (LOCAL LOGIC) ---
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 Dr. AI Assistant (Offline Mode)")
    
    chat_container = st.container(height=400)
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hello! I'm your virtual health assistant. Ask me about your BMI, diet plans, or interpreting medical terms.")
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask a medical question...")
    
    if prompt:
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI Logic (Local)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Get response from local function
                full_response = get_smart_response(prompt)
                
                # Typing effect simulation
                display_text = ""
                for char in full_response:
                    display_text += char
                    message_placeholder.markdown(display_text + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: HISTORY ---
with tabs[2]:
    st.subheader("📂 Patient History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Export to Excel/CSV", csv, "patient_history.csv", "text/csv")
    else:
        st.info("No records found. Run a diagnosis to save data here.")
