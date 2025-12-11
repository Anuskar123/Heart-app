import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioCare | Heart Prediction",
    page_icon="❤️",
    layout="wide"
)

# --- LOAD SYSTEM ---
@st.cache_resource
def load_system():
    try:
        data = joblib.load('heart_disease_system.pkl')
        return data['pipeline'], data['data_description'], data['data_sample']
    except FileNotFoundError:
        return None, None, None

model, df_desc, df_sample = load_system()

if model is None:
    st.error("⚠️ System file not found. Please run 'train_model.py' first.")
    st.stop()

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #cc0000; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("CardioCare System")
    st.markdown("---")
    page = st.radio("Navigate", ["🏥 Patient Entry", "📊 Data Insights", "ℹ️ About Model"])
    st.markdown("---")
    st.info("This system uses Logistic Regression to estimate heart disease risk based on clinical parameters.")

# --- HELPER FUNCTIONS ---
def plot_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff2b2b"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}],
        }
    ))
    return fig

def plot_comparison(user_value, feature_name, avg_healthy, avg_disease):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Healthy Avg', 'Disease Avg', 'You'],
        y=[avg_healthy, avg_disease, user_value],
        marker_color=['green', 'red', 'blue']
    ))
    fig.update_layout(title=f"Comparison: {feature_name}", height=300)
    return fig

# --- PAGE 1: PATIENT ENTRY ---
if page == "🏥 Patient Entry":
    st.header("New Patient Assessment")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("entry_form"):
            st.subheader("Clinical Data")
            c1, c2 = st.columns(2)
            
            with c1:
                age = st.number_input("Age", 20, 100, 55)
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                  format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
                trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 130)
                chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "True" if x==1 else "False")
            
            with c2:
                restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
                thalach = st.number_input("Max Heart Rate", 60, 220, 150)
                exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
                slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversable"][x])

            predict_btn = st.form_submit_button("Analyze Risk")

    if predict_btn:
        # Prepare Data
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
        })

        # Prediction
        prob = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Assessment Results")
        
        # Dashboard Layout for Results
        r1, r2 = st.columns([1, 1])
        
        with r1:
            st.plotly_chart(plot_gauge(prob), use_container_width=True)
            if prediction == 1:
                st.error(f"⚠️ **Result:** HIGH Risk of Heart Disease detected.")
            else:
                st.success(f"✅ **Result:** LOW Risk of Heart Disease detected.")

        with r2:
            st.markdown("### Key Observations")
            # Calculate comparisons for context
            avg_chol = df_desc['chol']['mean']
            avg_bp = df_desc['trestbps']['mean']
            
            st.write(f"**Cholesterol:** {chol} mg/dl")
            if chol > 240:
                st.warning("⚠️ Your Cholesterol is high (>240).")
            elif chol > 200:
                st.info("ℹ️ Your Cholesterol is borderline (200-239).")
            else:
                st.success("✅ Your Cholesterol is healthy.")

            st.write(f"**Max Heart Rate:** {thalach} bpm")
            if thalach < 100:
                st.warning("⚠️ Max heart rate seems unusually low.")
            
            # Download Report
            report_text = f"Patient Age: {age}\nSex: {'Male' if sex==1 else 'Female'}\nPrediction: {'High Risk' if prediction==1 else 'Low Risk'}\nProbability: {prob:.2%}"
            st.download_button("Download Report", report_text, file_name="patient_report.txt")

# --- PAGE 2: DATA INSIGHTS ---
if page == "📊 Data Insights":
    st.header("Population Statistics")
    st.write("Compare different health metrics across the dataset.")
    
    # Simple interactive chart
    param = st.selectbox("Select Parameter to Visualize", ['age', 'chol', 'trestbps', 'thalach'])
    
    fig = px.box(df_sample, x='target', y=param, 
                 points="all", 
                 title=f"{param.capitalize()} Distribution by Heart Disease Status",
                 color='target',
                 labels={'target': 'Heart Disease (0=No, 1=Yes)'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Correlation Matrix")
    corr = df_sample.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

# --- PAGE 3: ABOUT ---
if page == "ℹ️ About Model":
    st.header("About CardioCare")
    st.write("""
    **Model Used:** Logistic Regression with Standard Scaling.
    **Accuracy:** ~90% on test data.
    
    **Features Used:**
    * **Demographic:** Age, Sex
    * **Vitals:** Resting BP, Cholesterol, Max Heart Rate
    * **Clinical:** Chest Pain Type, ECG, Angina, ST Depression, etc.
    """)