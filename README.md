# 🚀 Hi!ckathon 2025 - PISA Score Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![AutoGluon](https://img.shields.io/badge/ML-AutoGluon-FFD700?logo=amazonsagemaker&logoColor=black)](https://auto.gluon.ai/)
[![Hi!Paris](https://img.shields.io/badge/Event-Hi!Paris_Hackathon-purple)](https://www.hi-paris.fr/)

> **Submission for the 6th edition of the Hi!ckathon (2025)**, organized by the Hi!Paris Center (IP Paris - HEC).

## 📄 Context & Objective
The goal of this project is to **predict mathematics scores in the international PISA test**. 
* **Challenge:** Analyze and model the impact of diverse factors on student performance using a large-scale dataset.
* **Key Variables:** Socio-economic background, science literacy, and reading/literal competencies.
* **Goal:** Identify the strongest predictors of academic success and build a robust regression model.

## ⚙️ Technical Pipeline
Our approach combined advanced feature representation with state-of-the-art AutoML frameworks.

### 1. Pre-processing & Feature Engineering (My Contribution)
* **Denoising AutoEncoder (DAE):** Implemented a DAE to learn a robust, compressed representation of the socio-economic features, effectively reducing noise and capturing non-linear correlations.
* **Data Cleaning:** Handled high-dimensional survey data, treated missing values through KNN imputation, and managed outliers.
* **Feature Engineering:** Integrated Science and Literacy scores while ensuring strict prevention of data leakage.

### 2. Modeling & Optimization
* **AutoGluon Framework:** Leveraged AutoGluon for automated model selection and multi-layer stacking, training an ensemble of diverse models (including CatBoost, LightGBM, and Neural Networks).
* **Validation:** Used repeated cross-validation to ensure the stability of PISA score predictions across different student demographics.
  
## 👥 The Team
Project realized in collaboration with students from Institut Polytechnique de Paris and HEC Paris.

## 📊 Results
* **Metric Performance:** RMSE
* **Final Score:** 0.77 (ranking top 10 on 70 groups) 

## 🛠️ Install & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/beltzantoine05/hickathon-6.git](https://github.com/beltzantoine05/hickathon-6.git)
   cd hickathon-6
