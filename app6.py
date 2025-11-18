import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import re
import time
from datetime import datetime

# ------------------------------------
# CONFIG & CSS (TRANSPARENT GLASS THEME)
# ------------------------------------
st.set_page_config(page_title="Credit Fraud Detection", page_icon="💳", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>
        /* 1. ANIMATED BACKGROUND */
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #020617, #1e1b4b, #312e81, #020617);
            background-size: 400% 400%;
            animation: gradientBG 20s ease infinite;
            color: #f8fafc;
        }

        /* 2. GLOBAL TEXT */
        h1, h2, h3, p, label, span, div {
            color: #f8fafc !important;
            font-family: 'Segoe UI', sans-serif;
            text-shadow: 0 0 10px rgba(0,0,0,0.8);
        }

        /* 3. TRANSPARENT INPUT FIELDS (User Request) */
        /* Makes the input box itself see-through */
        div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input {
            background-color: rgba(0, 0, 0, 0.2) !important; /* See-through dark tint */
            color: #ffffff !important; /* Bright white text */
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            font-weight: 500 !important;
            backdrop-filter: blur(5px);
        }
        
        div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus {
            border-color: #60a5fa !important;
            background-color: rgba(0, 0, 0, 0.4) !important;
            box-shadow: 0 0 15px rgba(96, 165, 250, 0.5) !important;
        }

        /* Labels above inputs */
        div[data-testid="stTextInput"] label, div[data-testid="stNumberInput"] label {
            color: #94a3b8 !important;
            font-size: 0.9rem !important;
            margin-bottom: 8px !important;
        }

        /* 4. GLASS CONTAINERS (Login Box) */
        .glass-container {
            background: rgba(255, 255, 255, 0.03); /* Very sheer */
            backdrop-filter: blur(20px); /* Heavy blur for readability */
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 50px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }

        /* 5. HUGE SELECTION BLOCKS & ANIMATIONS */
        
        /* Float Animation */
        @keyframes float {
            0% { transform: translateY(0px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            50% { transform: translateY(-20px); box-shadow: 0 25px 15px rgba(0,0,0,0.1); }
            100% { transform: translateY(0px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        }

        /* Border Glow Animation */
        @keyframes borderGlow {
            0% { border-color: rgba(255, 255, 255, 0.1); }
            50% { border-color: rgba(59, 130, 246, 0.6); }
            100% { border-color: rgba(255, 255, 255, 0.1); }
        }

        /* Target the main block buttons */
        div[data-testid="column"] .stButton button {
            height: 450px !important; /* MUCH BIGGER */
            width: 100% !important;
            background: rgba(255, 255, 255, 0.02) !important; /* Almost invisible glass */
            backdrop-filter: blur(10px) !important;
            border: 2px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 30px !important;
            
            /* Apply Animations */
            animation: float 6s ease-in-out infinite, borderGlow 4s ease-in-out infinite !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hover Effect - Stop float and glow intensely */
        div[data-testid="column"] .stButton button:hover {
            animation-play-state: paused !important;
            transform: scale(1.05) !important;
            background: rgba(59, 130, 246, 0.1) !important;
            border-color: #60a5fa !important;
            box-shadow: 0 0 50px rgba(59, 130, 246, 0.4) !important;
        }
        
        /* Big Text inside blocks */
        div[data-testid="column"] .stButton button p {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            background: -webkit-linear-gradient(#fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* 6. STANDARD BUTTONS */
        .stButton button {
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 15px 30px;
            font-size: 1rem;
            font-weight: bold;
            letter-spacing: 1.5px;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }
        
        /* 7. NAVBAR */
        .nav-bar {
            background: rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(15px);
            padding: 25px;
            border-radius: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 40px;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------
# HELPER: RERUN & MOCKING
# ------------------------------------
def safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def show_splash_screen():
    if "splash_shown" not in st.session_state:
        splash = st.empty()
        with splash.container():
            st.markdown("""
            <style>
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.2); opacity: 0.8; }
                100% { transform: scale(1); opacity: 1; }
            }
            </style>
            <div style="height: 90vh; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 8rem; animation: pulse 3s infinite;">🛡️</div>
                <h1 style="font-size: 4rem; font-weight: 900; color: #fff; letter-spacing: -2px;">SECURE GATEWAY</h1>
                <p style="font-size: 1.5rem; color: #94a3b8; letter-spacing: 5px; margin-top: 10px;">SYSTEM INITIALIZATION</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(2.5)
        splash.empty()
        st.session_state["splash_shown"] = True

def is_valid_luhn(card_number):
    card_number = re.sub(r'\D', '', card_number)
    if not card_number: return False
    n_digits = len(card_number)
    n_sum = 0
    is_second = False
    for i in range(n_digits - 1, -1, -1):
        d = int(card_number[i])
        if is_second: d = d * 2
        n_sum += d // 10
        n_sum += d % 10
        is_second = not is_second
    return n_sum % 10 == 0

def validate_expiry(expiry_str):
    try:
        if not re.match(r"^(0[1-9]|1[0-2])\/\d{2}$", expiry_str): return False, "Format MM/YY"
        exp_month, exp_year = map(int, expiry_str.split('/'))
        exp_year += 2000
        current_date = datetime.now()
        exp_date = datetime(exp_year, exp_month, 1)
        if exp_date < datetime(current_date.year, current_date.month, 1): return False, "Expired"
        return True, "Valid"
    except: return False, "Invalid Date"

class MockModel:
    def predict(self, data): return np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])
    def predict_proba(self, data):
        n = len(data)
        p1 = np.random.uniform(0, 1, n)
        p0 = 1 - p1
        return np.column_stack((p0, p1))

# ------------------------------------
# 1. AUTHENTICATION PAGE
# ------------------------------------
def auth_page():
    c1, c2, c3 = st.columns([1, 1.2, 1])
    
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="font-size: 5rem; margin-bottom: 10px;">💳</div>
            <h2 style="font-weight: 900; font-size: 2.5rem; letter-spacing: -1px;">LOGIN REQUIRED</h2>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 ACCESS PORTAL", "📝 NEW IDENTITY"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                # The CSS forces these to be transparent with white text
                username = st.text_input("USERNAME", placeholder="Enter ID (e.g., admin)")
                password = st.text_input("PASSWORD", type="password", placeholder="Enter Key (e.g., 1234)")
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("AUTHENTICATE", use_container_width=True)

                if submitted:
                    users = st.session_state.get("users", {"admin": "1234"})
                    if username in users and users[username] == password:
                        st.session_state["logged_in"] = True
                        st.session_state["current_user"] = username
                        st.session_state["app_mode"] = "selection"
                        safe_rerun()
                    else:
                        st.error("⛔ ACCESS DENIED")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form"):
                email = st.text_input("EMAIL ADDRESS")
                new_user = st.text_input("DESIRED USERNAME")
                new_pass = st.text_input("CREATE PASSWORD", type="password")
                confirm_pass = st.text_input("CONFIRM PASSWORD", type="password")
                st.markdown("<br>", unsafe_allow_html=True)
                register = st.form_submit_button("GENERATE CREDENTIALS", use_container_width=True)

                if register:
                    if not email or not new_user or not new_pass: st.error("FIELDS INCOMPLETE")
                    elif new_pass != confirm_pass: st.error("PASSWORD MISMATCH")
                    elif "users" in st.session_state and new_user in st.session_state["users"]: st.error("ID UNAVAILABLE")
                    else:
                        if "users" not in st.session_state: st.session_state["users"] = {"admin": "1234"}
                        st.session_state["users"][new_user] = new_pass
                        st.success("✅ ID GENERATED. PROCEED TO LOGIN.")

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------
# 2. SELECTION PAGE (BIG ANIMATED BLOCKS)
# ------------------------------------
def selection_page():
    st.markdown(f"""
    <div class="nav-bar">
        <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -1px;">🛡️ FRAUD GUARD AI</div>
        <div style="background: rgba(255,255,255,0.1); padding: 8px 20px; border-radius: 30px; font-weight: bold;">OPERATOR: {st.session_state.get('current_user', 'Admin').upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; margin-bottom: 60px; opacity: 0.7; letter-spacing: 3px;'>SELECT OPERATION PROTOCOL</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # CSS makes this 450px tall and animated
        if st.button("⚡\n\nREAL-TIME SCAN", use_container_width=True):
            st.session_state["app_mode"] = "manual"
            safe_rerun()

    with col2:
        # CSS makes this 450px tall and animated
        if st.button("📂\n\nBATCH UPLOAD", use_container_width=True):
            st.session_state["app_mode"] = "batch"
            safe_rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 0.6, 1])
    with c2:
        if st.button("🔒 SECURE LOGOUT", use_container_width=True):
            st.session_state["logged_in"] = False
            safe_rerun()

# ------------------------------------
# 3. MANUAL CHECK
# ------------------------------------
def instant_check_page(model, model_columns):
    st.markdown('<div class="nav-bar"><div>⚡ REAL-TIME SCANNER</div></div>', unsafe_allow_html=True)
    
    if st.button("⬅️ RETURN TO DASHBOARD"):
        st.session_state["app_mode"] = "selection"
        safe_rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💳 CARD IDENTITY")
        card_number = st.text_input("CARD NUMBER", max_chars=19)
        card_name = st.text_input("HOLDER NAME")
        expiry = st.text_input("EXPIRY (MM/YY)", max_chars=5)
        cvv = st.text_input("CVV", type="password", max_chars=4)
    with c2:
        st.markdown("#### 💸 TRANSACTION DATA")
        amount = st.number_input("AMOUNT ($)", value=100.0)
        time_val = st.number_input("TIMESTAMP (sec)", value=0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("INITIATE DIAGNOSTIC SCAN", use_container_width=True):
        with st.spinner("DECRYPTING & ANALYZING..."): time.sleep(1.2)
        
        errors = []
        if not is_valid_luhn(card_number): errors.append("INVALID LUHN CHECKSUM")
        valid_date, msg = validate_expiry(expiry)
        if not valid_date: errors.append(msg.upper())
        
        if errors:
            for e in errors: st.error(e)
        else:
            data = pd.DataFrame([{"Amount": amount, "Time_Hour": 0}])
            df_proc = pd.DataFrame(columns=model_columns)
            for c in model_columns: df_proc.loc[0, c] = 0
            df_proc["Amount"] = amount
            
            pred = model.predict(df_proc)[0]
            prob = model.predict_proba(df_proc)[0][1]
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
            if pred == 1:
                st.error(f"🚨 FRAUD CONFIRMED (RISK: {prob:.2%})")
            else:
                st.success(f"✅ TRANSACTION VERIFIED (SAFE: {1-prob:.2%})")
                st.balloons()

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------
# 4. BATCH CHECK
# ------------------------------------
def batch_upload_page(model, model_columns):
    st.markdown('<div class="nav-bar"><div>📂 BATCH PROCESSOR</div></div>', unsafe_allow_html=True)
    
    if st.button("⬅️ RETURN TO DASHBOARD"):
        st.session_state["app_mode"] = "selection"
        safe_rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("UPLOAD LOG FILE (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head(), use_container_width=True)
        if st.button("EXECUTE BATCH ANALYSIS", use_container_width=True):
            st.success("ANALYSIS COMPLETE - NO THREATS FOUND (MOCK)")
            st.balloons()
            
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------
# MAIN
# ------------------------------------
def main():
    load_custom_css()
    show_splash_screen()
    
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if "app_mode" not in st.session_state: st.session_state["app_mode"] = "selection"
    if "users" not in st.session_state: st.session_state["users"] = {"admin": "1234"}

    try:
        model = joblib.load("fraud_model.joblib")
        model_columns = joblib.load("model_columns.joblib")
    except:
        model = MockModel()
        model_columns = ["Amount", "Time_Hour"]

    if not st.session_state["logged_in"]:
        auth_page()
    else:
        if st.session_state["app_mode"] == "selection": selection_page()
        elif st.session_state["app_mode"] == "manual": instant_check_page(model, model_columns)
        elif st.session_state["app_mode"] == "batch": batch_upload_page(model, model_columns)

if __name__ == "__main__":
    main()