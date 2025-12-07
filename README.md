# 🌟 Non-Invasive Hemoglobin (Hb) Prediction from Fingertip Video

> **AI + Signal Processing + FastAPI Service**

This project builds a complete end-to-end machine learning pipeline that predicts **Hemoglobin (Hb)** levels from fingertip videos combined with basic demographic data.

---

## ✨ Features

- 📹 **Video → PPG extraction** using optical signal processing
- 🔬 **Rich PPG feature engineering** (time, frequency, physiological domains)
- 🤖 **ML training pipeline** with Ridge + RandomForest Ensemble
- 💻 **CLI prediction script** for quick analysis
- 🌐 **Fully functional FastAPI server** for real-time API-based predictions

---

## 🧠 How It Works — Overview

```
Fingertip Video (.mp4)
        ↓
Extract GREEN Signal
        ↓
Build PPG Time-Series
        ↓
Rich Feature Set
(time-domain + frequency + HR)
        ↓
Machine Learning Model
(Ridge + RandomForest)
        ↓
   Predicted Hb
```

---

## 📂 Project Structure

```
project/
│
├─ train_data/              # Labeled training videos (Hb in filename)
│  ├─ male_25_70_14.6.mp4
│  ├─ female_23_53_11.3.mp4
│
├─ test_data/               # Test videos (no Hb value)
│  ├─ male_24_45.mp4
│
├─ train_data_ppg/          # Auto-generated PPG CSVs
│
├─ hb_model.joblib          # Saved model (after training)
│
├─ extract_ppg.py           # Video → PPG CSV for training
├─ train_model.py           # PPG CSV → ML model
├─ predict_hb.py            # Predict Hb for a single video
├─ api_app.py               # FastAPI-based prediction API
└─ README.md
```

---

## 🏷️ Filename Format Requirements

### Training Videos

Must include true Hb label in filename:

**Format:** `gender_age_weight_hb.mp4`

**Examples:**

- `male_25_70_14.6.mp4`
- `female_23_53_11.3.mp4`

### Testing Videos

**Format:** `gender_age_weight.mp4`

**Examples:**

- `male_24_45.mp4`
- `female_30_62.mp4`

---

## ⚙️ Environment Setup (Windows / PowerShell)

### 1. Navigate to Project Directory

```powershell
cd "C:\Path\To\Project"
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment

```powershell
.\venv\Scripts\activate
```

### 4. Install Dependencies

```powershell
pip install opencv-python numpy pandas scipy scikit-learn joblib fastapi uvicorn[standard]
```

---

## 🔬 Step 1 — Extract PPG from Training Videos

**Script:** `extract_ppg.py`

### What it does:

- Extracts mean **GREEN intensity** from center 50% × 50% ROI
- Builds a time-series signal
- Saves CSV to `train_data_ppg/`

### Run:

```powershell
python extract_ppg.py
```

### Output:

PPG CSVs will appear in:

```
train_data_ppg/
├─ male_25_70_14.6.csv
├─ female_23_53_11.3.csv
```

---

## 🤖 Step 2 — Train the Hb Prediction Model

**Script:** `train_model.py`

### What training does:

1. Loads all CSVs from `train_data_ppg/`
2. Extracts **rich PPG features**:

| Category             | Features                                       |
| -------------------- | ---------------------------------------------- |
| **Statistics**       | mean, std, min, max, p25, p50, p75, range, IQR |
| **Shape**            | skew, kurtosis                                 |
| **Derivative**       | diff_mean, diff_std, diff_abs                  |
| **Autocorrelation**  | lag1, lag2                                     |
| **Frequency domain** | band powers (low/mid/high), power ratios       |
| **Physiological**    | estimated heart rate                           |
| **Signal**           | power, energy                                  |
| **Metadata**         | gender, age, weight                            |

3. Cross-validates **Ridge** (alpha ∈ {0.1, 1, 10})
4. Cross-validates **RandomForest** (max_depth ∈ {3, 4, 5, 6, None})
5. Builds **Ensemble** = 0.5 × Ridge + 0.5 × RF
6. Saves model to `hb_model.joblib`

### Run:

```powershell
python train_model.py
```

### Output:

You will see printed:

- ✅ CV MAE (realistic error)
- 📊 Train MAE (debug)
- 💾 Saved model path

---

## 🎯 Step 3 — Predict Hb for a Single Video (CLI)

**Script:** `predict_hb.py`

### Predict from command line:

```powershell
python predict_hb.py --video test_data\male_24_45.mp4
```

### Output:

```
=========================================
Predicted Hb for test_data\male_24_45.mp4: 13.82 g/dL
=========================================
```

---

## 🌐 Step 4 — FastAPI HTTP Service

**Script:** `api_app.py`

### Start the API:

```powershell
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000
```

### Open documentation:

🔗 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoint: `POST /predict`

### Request Type

`multipart/form-data`

### Fields:

| Field    | Type   | Description            |
| -------- | ------ | ---------------------- |
| `gender` | text   | `male` or `female`     |
| `age`    | number | years                  |
| `weight` | number | kg                     |
| `video`  | file   | `.mp4` fingertip video |

### Response:

```json
{
  "hb_pred": 13.7,
  "hr_bpm": 78.2,
  "duration_sec": 14.97,
  "num_samples": 450,
  "gender": "male",
  "age": 24.0,
  "weight": 70.0
}
```

---

## 🖥️ Calling the API from PowerShell

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post `
  -Form @{
    gender = "male"
    age = "24"
    weight = "70"
    video = Get-Item ".\test_data\male_24_45.mp4"
  }
```

---

## 🚀 Quick Commands Summary

### Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python numpy pandas scipy scikit-learn joblib fastapi uvicorn[standard]
```

### Extract PPG

```powershell
python extract_ppg.py
```

### Train Model

```powershell
python train_model.py
```

### Predict (CLI)

```powershell
python predict_hb.py --video test_data\sample.mp4
```

### Run API

```powershell
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📈 Accuracy Notes

- **Cross-validation MAE** is your true model accuracy
- More training samples → **significantly improved accuracy**
- Keep videos consistent:
  - ✅ Steady hand
  - ✅ Strong uniform lighting
  - ✅ Full fingertip coverage
  - ✅ Camera not moving

---

## 📝 License

This project is open-source and available for educational and research purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📧 Contact

For questions or support, please open an issue in the repository.

---

# How to Run Realtime PPG Scrip

## Webcam

```
python realtime_ppg.py
```

## Or with video file

```
python realtime_ppg.py --video test_data/male_25_70_14.6.mp4 --fs 60
```

**Made with ❤️ for non-invasive health monitoring**
