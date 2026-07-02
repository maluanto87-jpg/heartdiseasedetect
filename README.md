# ❤️ Heart Disease Prediction App

A simple web app that predicts the likelihood of heart disease from patient health data, using a Logistic Regression model trained on the UCI Heart Disease dataset. Built with **Streamlit** and **scikit-learn**.

🔗 **Live demo:**https://heartdiseasedetect-ecgvhayfyc6fud6qgmtiqo.streamlit.app/

---

## 📋 Overview

This app takes 13 clinical features (age, blood pressure, cholesterol, chest pain type, etc.) as input and predicts whether a patient is likely to have heart disease, along with a probability score.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is **not** a medical diagnostic tool and should never be used as a substitute for professional medical advice.

---

## 🗂️ Project Structure

```
heart/
├── app.py              # Streamlit web app
├── train_model.py       # Script to train and save the model
├── heart_model.pkl      # Pre-trained Logistic Regression model
├── heart.csv             # Dataset (UCI Heart Disease)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🧠 Model

- **Algorithm:** Logistic Regression (scikit-learn)
- **Dataset:** [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (1025 rows, 13 features)
- **Features used:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target:** `target` (1 = heart disease present, 0 = no heart disease)

| Metric | Score |
|---|---|
| Accuracy | ~0.80 |
| Precision | ~0.76 |
| Recall | ~0.87 |
| F1-score | ~0.81 |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> On Windows, if `pip` isn't recognized, use `py -m pip install -r requirements.txt` instead.

### 3. (Optional) Retrain the model
A pre-trained `heart_model.pkl` is already included. To retrain it yourself (e.g. after modifying the dataset or pipeline):
```bash
python train_model.py
```

### 4. Run the app locally
```bash
streamlit run app.py
```
> On Windows: `py -m streamlit run app.py`

The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (must be public, or you're on a plan that supports private repos).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select this repo, branch, and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs everything from `requirements.txt` automatically.
5. Copy the generated URL and share it, or add it to the top of this README.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — model training
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data handling

---

## 📌 Notes

- The model is trained on **raw (unscaled)** features to match the original notebook pipeline — no `StandardScaler` is applied.
- The dataset contains some duplicate rows (a known property of this commonly-circulated version of the UCI dataset); consider deduplicating in `train_model.py` if you want a stricter evaluation.

---
