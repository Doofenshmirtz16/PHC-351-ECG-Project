import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
from keras.models import load_model # Import load_model explicitly

# --- Configuration & Model Loading ---

# 🛑 🛑 🛑 MANDATORY: REPLACE THE PATH BELOW WITH YOUR ABSOLUTE PROJECT ROOT PATH 🛑 🛑 🛑
# Example: r'C:\Users\YourName\Desktop\ECG_Project'
PROJECT_ROOT = r'D:\ECG_Project' 
# 🛑 🛑 🛑 MANDATORY: REPLACE THE PATH ABOVE WITH YOUR ABSOLUTE PROJECT ROOT PATH 🛑 🛑 🛑

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FS = 360  # Sampling rate must match training data
WINDOW_SIZE = 180


@st.cache_resource
def load_models():
    # 1. Load Deep Learning Model (CNN-LSTM)
    dl_model = None
    try:
        # Use the explicit load_model import
        dl_model = load_model(os.path.join(MODEL_DIR, 'cnn_lstm_final_interpatient.keras'), compile=False)
    except Exception as e:
        st.error(f"❌ Error loading DL model (cnn_lstm_final_interpatient.keras): {e}")

    # 2. Load Classical ML Model (SVM)
    ml_model = None
    try:
        ml_model = joblib.load(os.path.join(MODEL_DIR, 'SVM_interpatient.joblib'))
    except Exception as e:
        st.error(f"❌ Error loading ML model (SVM_interpatient.joblib): {e}")

    # 3. Load Scaler for Classical ML (CRITICAL for feature normalization)
    scaler = None
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'StandardScaler_interpatient.joblib'))
    except Exception as e:
        st.warning(f"⚠️ Warning: Could not load StandardScaler. Classical ML features will be unscaled. ({e})")

    return dl_model, ml_model, scaler

# Helper function to extract the same features used in Week 3/5
def extract_features(segment_array):
    features = []
    for seg in segment_array:
        seg = seg.flatten()
        
        # Time Domain Features
        mean_val = np.mean(seg)
        std_val = np.std(seg)
        skew_val = skew(seg)
        kurt_val = kurtosis(seg)
        max_val = np.max(seg)
        min_val = np.min(seg)
        ptp_val = max_val - min_val
        energy_val = np.sum(seg**2)
        
        # Frequency Domain Features
        N = len(seg)
        freq_spectrum = np.abs(np.fft.rfft(seg))
        freq_energy = np.sum(freq_spectrum**2)
        dominant_freq_idx = np.argmax(freq_spectrum)
        normalized_spectrum = freq_spectrum / (np.sum(freq_spectrum) + 1e-9) 
        spec_entropy = entropy(normalized_spectrum)
        
        features.append([mean_val, std_val, skew_val, kurt_val, max_val, min_val, ptp_val, energy_val, freq_energy, dominant_freq_idx, spec_entropy])
    
    col_names = ["mean", "std", "skew", "kurtosis", "max", "min", "ptp", "energy", "freq_energy", "dominant_freq_idx", "spectral_entropy"]
    return pd.DataFrame(features, columns=col_names)


# --- Core Prediction Logic ---

def predict_abnormality(ecg_signal, dl_model, ml_model, scaler):
    # PREPROCESSING: Filter and find R-peaks
    try:
        signal_clean = nk.ecg_clean(ecg_signal, sampling_rate=FS, method='neurokit', highcut=40, lowcut=0.5)
        _, rpeaks = nk.ecg_peaks(signal_clean, sampling_rate=FS, method='pantompkins1985')
        rpeak_indices = rpeaks['ECG_R_Peaks']
    except Exception as e:
        st.error(f"Preprocessing Error (Filtering/R-peak): {e}")
        return None, None, None

    if len(rpeak_indices) < 1:
        st.warning("No R-peaks detected in the sample. Cannot proceed with beat segmentation.")
        return None, None, None

    # SEGMENTATION: Center around the first R-peak
    idx = rpeak_indices[0]
    if idx - WINDOW_SIZE // 2 < 0 or idx + WINDOW_SIZE // 2 >= len(signal_clean):
        st.warning("First R-peak too close to signal edges. Try a longer sample.")
        return None, None, None

    segment = signal_clean[idx - WINDOW_SIZE // 2 : idx + WINDOW_SIZE // 2]
    
    # NORMALIZATION (Per Segment - CRITICAL)
    segment_norm = (segment - np.mean(segment)) / np.std(segment)
    
    # 1. DL Prediction (CNN-LSTM)
    X_dl = segment_norm.reshape(1, WINDOW_SIZE, 1)
    dl_proba = dl_model.predict(X_dl, verbose=0)[0, 0]
    dl_pred = "Abnormal" if dl_proba >= 0.5 else "Normal"

    # 2. ML Prediction (SVM)
    X_ml_raw_feat = extract_features(np.array([segment_norm]))
    
    X_ml_scaled_feat = X_ml_raw_feat.values 
    if scaler:
        # Scaling requires feature names in the dataframe, convert back to values after scaling
        X_ml_scaled_feat = scaler.transform(X_ml_raw_feat)
    
    ml_pred = ml_model.predict(X_ml_scaled_feat)[0]
    ml_pred_label = "Abnormal" if ml_pred == 1 else "Normal"
    
    return dl_pred, ml_pred_label, segment_norm


# --- Streamlit UI ---

st.set_page_config(page_title="ECG Abnormality Detector", layout="wide")

st.title("🫀 AI-Based ECG Abnormality Detection Demo (Tachy/Brady)")
st.markdown("---")

# Load models and check dependencies
dl_model, ml_model, scaler = load_models()

# Check if required files were found 
if dl_model is None or ml_model is None:
    st.error("🛑 One or more models failed to load. Please verify: \n1. **ABSOLUTE PATH** in `run_demo.py` is correct. \n2. You ran the entire notebook successfully in the **Python 3.10 environment** to save all artifacts.")
    st.stop()


# --- Demo Input (Using a test patient) ---

st.header("1. Input ECG Data (Record 104 - Test Set Sample)")
st.info("We load the first 5 seconds of MIT-BIH record 104 (an unseen patient) to simulate a new reading. The model predicts the status of the first detected heartbeat.")

RECORD_TO_LOAD = '104'
SAMPLE_DURATION_SEC = 5 # Load 5 seconds

@st.cache_data
def load_sample_ecg(record_name):
    try:
        # Load the record using the ABSOLUTE DATA PATH
        record_path = os.path.join(DATA_DIR, record_name)
        # wfdb uses the file name without extension, but requires the path
        # Assuming the MITDB files are directly inside the DATA_DIR
        record = wfdb.rdrecord(record_path, sampto=FS * SAMPLE_DURATION_SEC) 
        return record.p_signal[:, 0]
    except Exception as e:
        st.error(f"Failed to load ECG data from WFDB path: {e}")
        return None

ecg_raw_signal = load_sample_ecg(RECORD_TO_LOAD)

if ecg_raw_signal is not None:
    # --- Run Prediction ---
    with st.spinner('Analyzing heartbeat and predicting abnormality...'):
        dl_pred, ml_pred_label, segment_norm = predict_abnormality(ecg_raw_signal, dl_model, ml_model, scaler)

    st.header("2. Prediction Results")

    col1, col2 = st.columns(2)

    if dl_pred and ml_pred_label:
        with col1:
            st.subheader("Deep Learning (CNN-LSTM)")
            dl_color = "green" if dl_pred == "Normal" else "red"
            st.markdown(f"**Prediction:** <span style='font-size: 24px; color: {dl_color}'>**{dl_pred}**</span>", unsafe_allow_html=True)
            st.caption("Trained on raw segment data.")

        with col2:
            st.subheader("Classical ML (SVM)")
            ml_color = "green" if ml_pred_label == "Normal" else "red"
            st.markdown(f"**Prediction:** <span style='font-size: 24px; color: {ml_color}'>**{ml_pred_label}**</span>", unsafe_allow_html=True)
            st.caption("Trained on statistical and spectral features.")

        st.markdown("---")

        # --- Visualization ---
        st.subheader(f"3. Visualized Heartbeat Segment (180 Samples)")
        st.caption("The colored line shows the single segment extracted and normalized for the prediction.")
        
        if segment_norm is not None:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(segment_norm, color='blue', linewidth=1)
            ax.axvline(x=WINDOW_SIZE/2, color='red', linestyle='--', label='R-peak Center', linewidth=2)
            ax.set_title("Normalized Beat Segment")
            ax.set_xlabel("Time Samples")
            ax.set_ylabel("Amplitude (Z-score)")
            ax.legend()
            st.pyplot(fig)