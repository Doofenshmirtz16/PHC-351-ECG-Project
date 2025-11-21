# 🩺 AI-Based Detection of Cardiovascular Abnormalities from ECG Signals

## Project Overview

This project implements a complete machine learning (ML) and deep learning (DL) pipeline to classify single-beat **Electrocardiogram (ECG) segments** for the presence of **abnormalities** (specifically targeting the detection of tachycardia and bradycardia related patterns).

The core goal was to demonstrate **clinical generalization** by strictly adhering to an **inter-patient data split** and providing **model interpretability** via SHAP and Grad-CAM.

| Feature | Status | Best Model Performance (Unseen Patients) |
| :--- | :--- | :--- |
| **Data Split** | Inter-Patient | Implemented (Train: 100, 101, 102; Test: 103, 104) |
| **Classical ML** | Implemented (SVM, RF, LR) | **SVM Accuracy (Clean):** 93.08% |
| **Deep Learning** | Implemented (CNN, LSTM, CNN-LSTM) | **CNN-LSTM AUC:** ~0.84 |
| **Interpretability** | SHAP (SVM) & Grad-CAM (CNN) | Successfully generated. |
| **Robustness** | Noise Injection Test | **Random Forest** showed superior stability (Accuracy: 91.38%). |
| **Final Deliverable** | **Streamlit Demo** | Working (Requires Python 3.10/3.11 environment) |

***

## 🛠️ 1. Setup and Environment

### Prerequisites

You must use a stable Python environment (e.g., **Python 3.10 or 3.11**). If you are currently on Python 3.12 or 3.13, you **must** use `venv` or `conda` to create a compatible environment to avoid library conflicts (especially with TensorFlow and Streamlit).

### Environment Setup (Using `venv` as a robust alternative)

1.  **Navigate to Project Root:** Use `cd` to enter the folder containing your project files (`run_demo.py`, `models/`, `data/`).
2.  **Create and Activate:**
    ```bash
    # Create the environment
    python -m venv .venv
    
    # Activate (Windows PowerShell/Command Prompt)
    .venv\Scripts\activate
    
    # Activate (Mac/Linux/Bash)
    source .venv/bin/activate
    ```

3.  **Install Dependencies:** Install all necessary packages into the active environment:
    ```bash
    pip install wfdb numpy scipy matplotlib pandas scikit-learn neurokit2 tensorflow joblib streamlit shap
    ```

***

## 2. Data Pipeline Methodology

The project is built around the **MIT-BIH Arrhythmia Database (`mitdb`)**. The entire analysis is centered around the **Inter-Patient Split** principle to ensure realistic model generalization.

### Core Preprocessing Steps

* **Filtering:** Applied **Bandpass filtering (0.5–40 Hz)** and baseline wander removal (via `neurokit2`).
* **R-Peak Detection:** Used to locate and center individual heartbeats.
* **Segmentation:** Beats segmented into fixed windows (**180 samples / 0.5s**).
* **Scaling:** **Z-score normalization** applied **per segment**.

***

## 3. Execution and Artifacts

### Final Analysis Notebook

The entire pipeline—from data loading to training, testing, and generating interpretability plots—is contained within the primary notebook. **It must be re-run in the stable environment.**

* **Final Notebook:** `Final_Notebook.ipynb`

### Critical Artifacts

The execution of the notebook successfully saves these files, which are essential for running the demo:

| Artifact | File Path | Source Notebook Cell |
| :--- | :--- | :--- |
| **Processed Data** | `../data/processed_full_interpatient.npz` | Cell 7 |
| **Feature Scaler** | `../models/StandardScaler_interpatient.joblib` | Cell 16 (Saving step added) |
| **Best ML Model (SVM)** | `../models/SVM_interpatient.joblib` | Cell 17 & 20 |
| **Best DL Model (CNN-LSTM)** | `../models/cnn_lstm_final_interpatient.keras` | Cell 34 |

***

## 4. Interpretability and Robustness (Week 5)

This phase validated the models' decisions and stability against simulated real-world conditions.

### Interpretability

* **SHAP Analysis (SVM):** Confirmed model reliance on **amplitude features** and **signal complexity features** (e.g., Spectral Entropy).
* **Grad-CAM (CNN/CNN-LSTM):** Visualized that the DL model focused primarily on the **QRS complex** for classification.

### Robustness Check (Noise Testing)

Gaussian noise (STD=0.1) was added to the unseen test set to simulate real-world signal artifacts.

| Model | Accuracy (Clean Test Set) | Accuracy (Noisy Test Set) | Conclusion |
| :--- | :--- | :--- | :--- |
| **SVM** | **93.08%** | **48.25%** | **Highly Fragile:** Performance collapsed when noise was introduced. |
| **Random Forest** | 89.95% | **91.38%** | **Highly Robust:** The ensemble nature maintained stability against noise. |

**Key Scientific Insight:** The **Random Forest** model, despite having lower raw accuracy than SVM on clean data, proved significantly more stable and reliable against simulated noise. This confirms that **robustness to real-world noise** is paramount and must be tested alongside traditional accuracy metrics.

***

## 5. Final Deliverables and Demo

### Running the Live Demo

The demo loads the trained models and runs a prediction on a sample from the unseen test set (Patient 104).

**CRITICAL:** You must be in the **project root directory** with the **stable environment activated** to execute this command:

```bash
streamlit run run_demo.py
```

### Created by:
Sumit Sharma
23123042
Engineering Physics
