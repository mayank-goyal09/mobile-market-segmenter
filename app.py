import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIG (Must be FIRST!)
# ==========================================
st.set_page_config(
    page_title="Mobile Market Segmenter",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. HELPER: GENERATE MODEL IF MISSING
# ==========================================
def generate_dummy_model():
    """
    Creates a dummy model and saves it to disk if it doesn't exist.
    This ensures the app runs immediately without needing an external file first.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # 1. Generate Fake Data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Age': np.random.randint(18, 70, n_samples),
        'App Usage Time (min/day)': np.random.randint(0, 600, n_samples),
        'Data Usage (MB/day)': np.random.randint(100, 5000, n_samples),
        'Battery Drain (mAh/day)': np.random.randint(1000, 4000, n_samples),
        'Screen On Time (hours/day)': np.random.uniform(1, 12, n_samples),
        'Number of Apps Installed': np.random.randint(10, 200, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Operating System': np.random.choice(['Android', 'iOS'], n_samples),
        'Device Model': np.random.choice(['iPhone 12', 'Samsung Galaxy S21', 'Xiaomi Mi 11'], n_samples)
    })

    # Target variable (Segments 1-5)
    y = np.random.randint(1, 6, n_samples)

    # 2. Preprocessing
    X = pd.get_dummies(data)
    model_columns = list(X.columns) # Save columns to ensure alignment later

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Train Model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y_encoded)

    # 4. Save Everything
    artifacts = {
        'model': model,
        'scaler': scaler,
        'le': le,
        'model_columns': model_columns
    }
    joblib.dump(artifacts, 'mobile_segmenter_model.pkl')
    return artifacts

# ==========================================
# 3. CUSTOM STYLING (Dark Green & Purple Theme)
# ==========================================
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 70%);
            color: #e0e0e0;
        }
        
        /* Headers */
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #ffffff;
        }
        
        .main-header {
            font-size: 3.5em;
            font-weight: 800;
            background: linear-gradient(90deg, #00ff88 0%, #bd00ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 0px 0px 20px rgba(0, 255, 136, 0.3);
            margin-bottom: 10px;
        }
        
        .sub-header {
            text-align: center;
            color: #a0a0a0;
            font-size: 1.2em;
            margin-bottom: 40px;
        }

        /* Glass Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(0, 255, 136, 0.3);
        }

        /* Prediction Box Styles */
        .pred-title {
            font-size: 1.1em;
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .pred-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }
        .highlight-green { color: #00ff88; text-shadow: 0 0 15px rgba(0, 255, 136, 0.4); }
        .highlight-purple { color: #bd00ff; text-shadow: 0 0 15px rgba(189, 0, 255, 0.4); }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0a0a0a;
            border-right: 1px solid #1f1f1f;
        }
        
        /* Custom Button */
        .stButton > button {
            background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
            color: black;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            height: 50px;
            font-size: 1.1em;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #00cc6a 0%, #00ff88 100%);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
            color: black;
        }

        /* Metrics */
        [data-testid="stMetricLabel"] { color: #888 !important; }
        [data-testid="stMetricValue"] { color: #fff !important; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    file_path = 'mobile_segmenter_model.pkl'
    
    # Check if model exists, if not, generate it!
    if not os.path.exists(file_path):
        st.toast("⚠️ Model not found. Generating new model...", icon="⚙️")
        return generate_dummy_model()
        
    try:
        artifacts = joblib.load(file_path)
        return artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================
def predict_segment(artifacts, input_data):
    model = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['le']
    model_columns = artifacts['model_columns']

    # One-Hot Encode
    input_dummies = pd.get_dummies(input_data)
    
    # CRITICAL FIX: Reindex matches columns to training data, filling missing cols with 0
    input_dummies = input_dummies.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_dummies)

    # Predict
    prediction_idx = model.predict(input_scaled)[0]
    prediction_name = str(le.inverse_transform([prediction_idx])[0])

    # Confidence
    proba = model.predict_proba(input_scaled).max()

    return prediction_name, proba


# ==========================================
# 6. BUSINESS INSIGHTS
# ==========================================
def get_business_insights(segment):
    # Ensure segment is string for dictionary lookup
    segment = str(segment)
    
    insights = {
        "1": {
            "emoji": "🧘",
            "name": "Light User",
            "desc": "Budget-conscious, minimal usage.",
            "plan": "Starter Pack (Talk & Text)",
            "upsell": "Family Bundles",
            "risk": "Low Engagement",
            "color": "#a0a0a0"
        },
        "2": {
            "emoji": "⚖️",
            "name": "Moderate User",
            "desc": "Balanced daily usage.",
            "plan": "Standard Bundle",
            "upsell": "Social Media Packs",
            "risk": "Medium Engagement",
            "color": "#00ccff"
        },
        "3": {
            "emoji": "🔥",
            "name": "Active User",
            "desc": "Frequent engagement & data use.",
            "plan": "Unlimited Data + Calls",
            "upsell": "Streaming Add-ons",
            "risk": "Low Risk",
            "color": "#00ff88"
        },
        "4": {
            "emoji": "🚀",
            "name": "Heavy User",
            "desc": "High performance & data needs.",
            "plan": "5G Premium Unlimited",
            "upsell": "Exclusive Content / Gaming",
            "risk": "Very Low Risk",
            "color": "#bd00ff"
        },
        "5": {
            "emoji": "⚡",
            "name": "Power User",
            "desc": "Extreme usage, always online.",
            "plan": "Elite Priority Tier",
            "upsell": "VIP Service / Enterprise",
            "risk": "Zero Risk",
            "color": "#ff0055"
        }
    }
    return insights.get(segment, insights["1"])


# ==========================================
# 7. RADAR CHART FUNCTION
# ==========================================
def plot_radar_chart(user_metrics):
    categories = ['App Time', 'Data Usage', 'Screen Time', 'Battery Drain', 'Apps Installed']
    
    # Normalized values for visualization (approximate scale 0-100)
    user_values = [
        (user_metrics[0] / 1440) * 100,  # App time
        (user_metrics[1] / 10000) * 100, # Data
        (user_metrics[2] / 24) * 100,    # Screen
        user_metrics[3],                 # Battery %
        (user_metrics[4] / 300) * 100    # Apps
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Current User',
        line_color='#00ff88',
        fillcolor='rgba(0, 255, 136, 0.2)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                gridcolor='#333'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff'),
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    return fig


# ==========================================
# 8. MAIN APP
# ==========================================
def main():
    
    # Header
    st.markdown('<div class="main-header">MOBILE NEXUS 📱</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Market Segmentation Interface</div>', unsafe_allow_html=True)

    # Load (or create) the model
    artifacts = load_model()

    # ------------------------------------------
    # SIDEBAR INPUT
    # ------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("👤 User Profile", expanded=True):
            age = st.slider("Age", 10, 100, 30)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        
        with st.expander("📱 Device Specs", expanded=True):
            device_map = {
                "Apple (iPhone)": ("iPhone 12", 2815),
                "Samsung": ("Samsung Galaxy S21", 4000),
                "Xiaomi": ("Xiaomi Mi 11", 4250),
                "Google (Pixel)": ("Google Pixel 5", 4080),
                "OnePlus": ("OnePlus 9", 4500)
            }
            device_choice = st.selectbox("Device Brand", list(device_map.keys()))
            device_model = device_map[device_choice][0]
            device_capacity = device_map[device_choice][1]
            
            # Smart OS Defaulting
            default_os_idx = 1 if "Apple" in device_choice else 0
            os_sys = st.radio("OS", ["Android", "iOS"], horizontal=True, index=default_os_idx)

        with st.expander("🔋 Usage Metrics", expanded=True):
            app_time = st.slider("App Time (min)", 0, 1440, 300)
            data_usage = st.slider("Data (MB)", 0, 10000, 1000)
            screen_time = st.slider("Screen Time (hrs)", 0.0, 24.0, 6.0)
            num_apps = st.slider("Apps Installed", 0, 300, 50)
            
            # Battery Logic
            battery_pct = st.slider("Daily Battery Drain %", 0, 100, 60)
            battery_mah = (battery_pct / 100) * device_capacity

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ ANALYZE USER SEGMENT")

    # ------------------------------------------
    # MAIN DASHBOARD
    # ------------------------------------------
    if predict_btn:
        # Prepare Input
        input_data = pd.DataFrame({
            'Age': [age],
            'App Usage Time (min/day)': [app_time],
            'Data Usage (MB/day)': [data_usage],
            'Battery Drain (mAh/day)': [battery_mah],
            'Screen On Time (hours/day)': [screen_time],
            'Number of Apps Installed': [num_apps],
            'Gender': [gender],
            'Operating System': [os_sys],
            'Device Model': [device_model]
        })

        segment, confidence = predict_segment(artifacts, input_data)
        insights = get_business_insights(segment)

        # --- Top Prediction Cards ---
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid {insights['color']};">
                    <div class="pred-title">USER CLASSIFICATION</div>
                    <div class="pred-value highlight-green">
                        {insights['emoji']} {insights['name']}
                    </div>
                    <div style="margin-top: 10px; color: #ccc; font-style: italic;">
                        "{insights['desc']}"
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="glass-card" style="border-left: 5px solid #bd00ff;">
                    <div class="pred-title">AI CONFIDENCE</div>
                    <div class="pred-value highlight-purple">
                        {confidence:.1%}
                    </div>
                    <div style="margin-top: 10px; color: #ccc;">
                        Probability Score
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # --- Strategic Insights ---
        st.markdown("### 🧠 Strategic Recommendations")
        col_strat1, col_strat2, col_strat3 = st.columns(3)

        with col_strat1:
            st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #00ff88; font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                        💎 Recommended Plan
                    </div>
                    <div style="font-size: 1.1em;">{insights['plan']}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_strat2:
            st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #bd00ff; font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                        🚀 Best Upsell
                    </div>
                    <div style="font-size: 1.1em;">{insights['upsell']}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_strat3:
            st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #ff4b4b; font-size: 1.2em; font-weight: bold; margin-bottom: 5px;">
                        ⚠️ Churn Risk
                    </div>
                    <div style="font-size: 1.1em;">{insights['risk']}</div>
                </div>
            """, unsafe_allow_html=True)

        # --- Visual Analytics ---
        st.markdown("### 📊 Behavior Analysis")
        
        viz_col1, viz_col2 = st.columns([1, 1.5])

        with viz_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("📱 App Usage", f"{app_time} min", delta="Daily Avg")
            st.metric("📶 Data Consumed", f"{data_usage} MB", delta="Daily Avg")
            st.metric("🔋 Battery Impact", f"{battery_pct}%", delta=f"{battery_mah:.0f} mAh")
            st.markdown('</div>', unsafe_allow_html=True)

        with viz_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<div class='pred-title' style='text-align: center;'>USER FOOTPRINT</div>", unsafe_allow_html=True)
            radar_fig = plot_radar_chart([app_time, data_usage, screen_time, battery_pct, num_apps])
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # --- Initial State ---
        st.markdown("""
            <div class="glass-card" style="text-align: center; margin-top: 50px; padding: 50px;">
                <h2 style="color: #00ff88;">Ready to Analyze?</h2>
                <p style="font-size: 1.2em; color: #ccc;">
                    👈 Configure the user profile in the sidebar and hit <b style="color: #bd00ff;">ANALYZE USER SEGMENT</b>
                    to generate AI predictions.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # --- Sticky Footer ---
    st.markdown(
    """
    <style>
    .footer-container {
        position: fixed; bottom: 0; left: 0; right: 0;
        padding-bottom: 12px; background: rgba(0,0,0,0.0);
    }
    .footer-icon:hover svg {
        filter: drop-shadow(0 0 6px rgba(0, 255, 180, 0.8));
        transform: scale(1.15); transition: 0.25s ease;
    }
    </style>
    <div class="footer-container">
    <hr style="border-color: rgba(255,255,255,0.15); margin-top: 1rem; width: 100%;"/>
    <div style="text-align: center; color: #e0e0e0; font-size: 0.9rem; padding-bottom: 1rem;">
        <div style="margin-bottom: 6px;">built by <b>Mayank's ML brain 🧠</b> · powered by Streamlit &amp; scikit-learn</div>
        <div style="margin-top: 4px; font-size: 1rem; font-weight: 600;">Mayank Goyal — Data Scientist</div>
        <div style="margin-top: 8px; display: flex; justify-content: center; gap: 18px;">
            <a class="footer-icon" href="https://www.linkedin.com/in/mayank-goyal-4b8756363/" target="_blank" style="text-decoration: none; color: #00a0dc;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#00a0dc" viewBox="0 0 16 16"><path d="M1.146 1.146c.4-.4 1.048-.4 1.447 0 .4.4.4 1.048 0 1.447-.4.4-1.048.4-1.447 0-.4-.4-.4-1.048 0-1.447zM0 4.5h3v11H0v-11zm5 0h2.837v1.558h.04c.396-.75 1.365-1.558 2.81-1.558C14.065 4.5 15 6.253 15 9.093V15.5h-3v-5.632c0-1.342-.027-3.066-1.868-3.066-1.87 0-2.156 1.46-2.156 2.966V15.5H5v-11z"/></svg>
            </a>
            <a class="footer-icon" href="https://github.com/mayank-goyal09" target="_blank" style="text-decoration: none; color: #fff;">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#ffffff" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
            </a>
        </div>
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
