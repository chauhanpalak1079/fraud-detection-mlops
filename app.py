import streamlit as st
import requests
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="centered"
)

# Title and description
st.title("🔒 Fraud Detection System")
st.markdown("### Analyze transactions for potential fraud")

# Load model directly (alternative to API)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fraud_detection_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for mode selection
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Select Mode:",
    ["Direct Model", "API Mode"],
    help="Direct Model uses the local model, API Mode calls the FastAPI endpoint"
)

if mode == "API Mode":
    api_url = st.sidebar.text_input(
        "API URL",
        value="http://localhost:8000/predict",
        help="Enter the FastAPI endpoint URL"
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 Enter transaction details to check for fraud")

# Main input form
st.markdown("#### Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f"
    )
    old_balance = st.number_input(
        "Old Balance ($)",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        format="%.2f"
    )

with col2:
    new_balance = st.number_input(
        "New Balance ($)",
        min_value=0.0,
        value=4000.0,
        step=100.0,
        format="%.2f"
    )

# Predict button
if st.button("🔍 Check for Fraud", type="primary", use_container_width=True):
    with st.spinner("Analyzing transaction..."):
        
        if mode == "Direct Model":
            # Use local model
            if model:
                features = np.array([[amount, old_balance, new_balance]])
                prediction = model.predict(features)
                probability = model.predict_proba(features)
                
                is_fraud = bool(prediction[0])
                confidence = float(np.max(probability)) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                if is_fraud:
                    st.error("🚨 **FRAUD DETECTED**")
                else:
                    st.success("✅ **Transaction Safe**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fraud Status", "FRAUD" if is_fraud else "SAFE")
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Additional details
                with st.expander("📋 Transaction Summary"):
                    st.write(f"**Amount:** ${amount:,.2f}")
                    st.write(f"**Old Balance:** ${old_balance:,.2f}")
                    st.write(f"**New Balance:** ${new_balance:,.2f}")
                    st.write(f"**Balance Change:** ${old_balance - new_balance:,.2f}")
            else:
                st.error("Model not loaded. Please ensure fraud_detection_model.pkl exists.")
        
        else:
            # Use API
            try:
                payload = {
                    "amount": amount,
                    "old_balance": old_balance,
                    "new_balance": new_balance
                }
                response = requests.post(api_url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    if result["is_fraud"]:
                        st.error(f"🚨 **{result['message']}**")
                    else:
                        st.success(f"✅ **{result['message']}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fraud Status", "FRAUD" if result["is_fraud"] else "SAFE")
                    with col2:
                        st.metric("Confidence", result["confidence_score"])
                    
                    # Additional details
                    with st.expander("📋 Transaction Summary"):
                        st.write(f"**Amount:** ${amount:,.2f}")
                        st.write(f"**Old Balance:** ${old_balance:,.2f}")
                        st.write(f"**New Balance:** ${new_balance:,.2f}")
                        st.write(f"**Balance Change:** ${old_balance - new_balance:,.2f}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI server is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Fraud Detection System | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
