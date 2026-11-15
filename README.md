# EduPulse AI — Student Academic Performance & Pass/Fail Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Deployment-GitHub%20Pages%20Ready-10B981?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-4F46E5?style=flat)]()

**EduPulse AI** is an interpretable Machine Learning pipeline and zero-backend web dashboard engineered to predict student academic outcomes (Pass / Fail) with **92.0% benchmark accuracy**. 

The system trains both **Logistic Regression** and **Random Forest** classifiers, generates **4 high-resolution (300 DPI)** analytical plots, serializes model weights to JSON, and delivers real-time, client-side inference in vanilla JavaScript running natively on **GitHub Pages** without requiring a Python server.

---

## 🌟 Key Features

- **⚡ Zero-Latency Browser Inference:** Model weights and normalization parameters run natively in JavaScript via matrix operations and Sigmoid evaluation with sub-millisecond ($<1\text{ ms}$) execution.
- **🎯 92.00% Benchmark Accuracy:** Stratified evaluation yielding **96.00% Precision** and an **0.923 F1-Score** on unseen test cases.
- **🔍 Explainable AI (XAI):** Real-time feature contribution bars break down exactly how each input parameter pushes the student towards *Pass* or *Risk*.
- **📊 300 DPI Academic Visualizations:** High-resolution charts styled with modern Indigo (`#4F46E5`) & Emerald (`#10B981`) palettes with white background and despined axes.
- **🚀 Static Cloud Deployment:** Requires zero cloud compute, zero database setup, and zero backend maintenance. Fully compatible with GitHub Pages, Vercel, and Netlify.

---

## 🧠 Machine Learning Performance Benchmark

The dataset was partitioned using an **80/20 stratified split** (`random_state=42`) with standard scaling fitted exclusively on training data to prevent data leakage.

| Evaluation Metric | Logistic Regression (L2) | Random Forest Classifier | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **92.00%** | 91.00% | $+1.00\%$ |
| **Precision** | **96.00%** | 92.45% | $+3.55\%$ |
| **Recall** | 88.89% | **90.74%** | $-1.85\%$ |
| **F1-Score** | **0.923** | 0.916 | $+0.007$ |
| **Confusion Matrix** | $TN=44,\; FP=2$<br>$FN=6,\; TP=48$ | $TN=42,\; FP=4$<br>$FN=5,\; TP=49$ | $+2\text{ True Negatives}$ |
| **Inference Engine** | Client-Side JS (Active) | Benchmark Baseline | Real-Time |

> **Key Pedagogical Finding:** Logistic Regression demonstrated superior precision ($96.0\%$) with only **2 false positives** across 100 test samples, minimizing false reassurance for students who actually require remedial academic intervention.

---

## 📋 Dataset Schema (`student_data.csv`)

The dataset comprises **500 student records** generated with realistic academic distributions and calibrated human performance variability ($\sigma = 0.040$).

| Feature Name | Type | Domain / Range | Mean ($\mu$) | Std ($\sigma$) | Description & Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Hours_Studied` | Float | $1.0\text{--}10.0\text{ hrs/wk}$ | $5.27$ | $2.62$ | Dedicated individual self-study time per week. |
| `Attendance_Percentage` | Float | $50.0\%\text{--}100.0\%$ | $74.55$ | $14.11$ | Classroom presence and active participation rate. |
| `Previous_Score` | Float | $30.0\text{--}100.0\text{ pts}$ | $66.12$ | $19.70$ | Historical academic baseline score. |
| `Assignments_Completed` | Integer | $0\text{--}10\text{ units}$ | $4.86$ | $3.25$ | Formative assessment completion rate. |
| **`Result` (Target)** | Binary | $0\text{ (Fail)} / 1\text{ (Pass)}$ | $54.0\%$ | $46.0\%$ | Final verified academic outcome. |

---

## 📈 High-DPI Visualizations (`outputs/`)

All charts are saved at **300 DPI** using Matplotlib with modern styling:

1. **[confusion_matrix.png](outputs/confusion_matrix.png)**: Side-by-side classification heatmaps showing counts, percentages, and class boundaries.
2. **[model_comparison.png](outputs/model_comparison.png)**: Grouped comparative bar chart evaluating Accuracy, Precision, Recall, and F1.
3. **[feature_importance.png](outputs/feature_importance.png)**: Quantitative feature ranking confirming `Previous_Score` ($35.9\%$) and `Hours_Studied` ($30.7\%$) as dominant drivers of success.
4. **[study_vs_score_scatter.png](outputs/study_vs_score_scatter.png)**: Bivariate scatter plot illustrating the empirical passing frontier and the **effort compensation effect**.

---

## 🔬 Mathematical Formulation for Browser Inference

The web app in `index.html` implements exact mathematical vector operations:

1. **Standardization (StandardScaler):**
   $$z_i = \frac{x_i - \mu_i}{\sigma_i}$$

2. **Log-Odds Dot Product (Logit):**
   $$\text{logit} = \beta_0 + \sum_{i=1}^4 \beta_i \cdot z_i$$
   *Exported Coefficients:* $\beta_{\text{hours}} = 2.7546$, $\beta_{\text{att}} = 2.2025$, $\beta_{\text{score}} = 2.9418$, $\beta_{\text{assign}} = 1.5887$, $\beta_0 = 0.4358$.

3. **Sigmoid Activation Function:**
   $$P(\text{Pass}) = \frac{1}{1 + e^{-\text{logit}}}, \quad P(\text{Fail}) = 1 - P(\text{Pass})$$

---

## 📁 Project Directory Structure

```
EduPulse-AI/
├── outputs/                                # 300 DPI High-Resolution Visualizations
│   ├── confusion_matrix.png                # Side-by-side model confusion matrices
│   ├── feature_importance.png              # Relative feature weight rankings
│   ├── model_comparison.png                # Comparative evaluation bar chart
│   └── study_vs_score_scatter.png          # Academic distribution & passing frontier
├── index.html                              # Interactive SaaS Dashboard (HTML/CSS/JS)
├── model_data.json                         # Serialized weights, scaler & metrics
├── student_data.csv                        # 500-sample academic dataset
├── train_model.py                          # ML pipeline & visualization generator
├── requirements.txt                        # Python dependencies
├── PPT_Outline.md                          # 8-slide presentation outline
└── README.md                               # Project documentation & reference
```

---

## 🚀 Quickstart & Execution Guide

### Option 1: Direct File Launch (No Setup Required)
Simply double-click `index.html` in Windows Explorer or open it in any modern browser. The application includes embedded fallback weights ensuring full offline functionality.

### Option 2: Run ML Training Pipeline
To regenerate the dataset, train models, generate 300 DPI charts, and export JSON weights:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute training pipeline
python train_model.py
```

### Option 3: Local HTTP Web Server
To test dynamic JSON fetching locally:

```bash
python -m http.server 8000
```
Then visit: `http://localhost:8000`

---

## 🌐 Deploy to GitHub Pages (1-Click Hosting)

1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: EduPulse AI static web app"
   ```
2. Create a repository on GitHub and push your branch:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/edupulse-ai.git
   git push -u origin main
   ```
3. Navigate to **Settings > Pages** in your GitHub repository:
   - **Source:** Deploy from a branch
   - **Branch:** `main` / `/ (root)`
   - Click **Save**.
4. Your dashboard will be live at `https://<your-username>.github.io/edupulse-ai/`.

---

## 🎓 Viva & Technical Defense Highlights

- **Q: Why was StandardScaler fit only on `X_train`?**  
  *A:* Fitting on the full dataset causes **data leakage**, allowing test distribution parameters to bleed into model training. Fitting solely on `X_train` ensures test evaluation accurately measures generalization on unseen profiles.
- **Q: Why does Logistic Regression outperform Random Forest here?**  
  *A:* Academic effort and aptitude have a continuous, monotonic relationship with success. Logistic Regression models the linear log-odds frontier smoothly, whereas tree models approximate linear boundaries with stepped orthogonal partitions.
- **Q: What is the Effort Compensation Effect?**  
  *A:* As visualized in `study_vs_score_scatter.png`, an increase of $\sim 2.5\text{ study hours/week}$ compensates for a 15-point deficit in previous academic scores.

---

## 📄 License
This project is released under the **MIT License**. Free for academic, institutional, and research use.
