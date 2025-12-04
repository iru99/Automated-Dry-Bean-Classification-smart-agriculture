# Automated Varietal Purity Certification in Smart Agriculture 🌱

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
This project addresses the critical challenge of **food fraud** and **varietal purity** in the global seed industry. By leveraging Machine Learning and Computer Vision data, we developed an automated optical sorting pipeline to classify seven registered varieties of dry beans (*Phaseolus vulgaris* L.).

The study critically compares complex **Heterogeneous Ensemble Architectures** against **Optimized Single Models** to determine the most efficient solution for real-time industrial deployment (Edge AI).

## 📊 Dataset
**Source:** [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)  
**Original Study:** Koklu, M. & Ozkan, I.A. (2020). DOI: [10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)

* **Instances:** 13,611
* **Features:** 16 Morphological/Geometric attributes (Area, Perimeter, ShapeFactors, etc.)
* **Classes:** 7 (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira)

## 🛠️ Methodology & Tech Stack
The pipeline was implemented using **Python** and **Scikit-Learn** within a Jupyter Notebook environment.

### 1. Data Preprocessing
* **Class Balancing:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to mitigate bias against minority classes like *Bombay*.
* **Dimensionality Reduction:** Used **Principal Component Analysis (PCA)** to reduce 16 multicollinear features to 6 components (retaining 95% variance).
* **Scaling:** Applied `StandardScaler` for distance-sensitive algorithms.

### 2. Algorithms Evaluated
* **K-Nearest Neighbors (KNN)** (Baseline)
* **Support Vector Machines (SVM)** (Optimized via GridSearchCV)
* **XGBoost** (Gradient Boosting)
* **Random Forest** (Regularized & Tuned)
* **Ensemble Voting Classifier** (Soft Voting aggregation of RF + XGB + SVM)

## 📈 Key Results
Contrary to the initial hypothesis that the Ensemble would perform best, the **Tuned Random Forest** emerged as the superior solution for industrial application.

| Model Architecture | Accuracy | F1-Score | Training Time (s) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest (Tuned)** | **89.13%** | **0.89** | **2.70s** | 🏆 **Optimal** |
| Ensemble Voting | 88.84% | 0.89 | 24.99s | Too Slow |
| XGBoost | 88.40% | 0.88 | 0.49s | Good |
| SVM (Optimized) | 85.97% | 0.86 | 21.96s | Weakest |

### 🧬 Biological Insight
Feature Importance analysis revealed that **ShapeFactor3** and **Compactness** were the most significant predictors. The model successfully "learned" that the *proportional geometry* of a seed is a more reliable genetic marker than its absolute size (*Area*).

## 🚀 How to Run
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/Automated-Dry-Bean-Classification.git](https://github.com/yourusername/Automated-Dry-Bean-Classification.git)
    ```
2.  **Install dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl
    ```
3.  **Run the Notebook**
    * Open `Dry_Bean_Classification.ipynb` in Jupyter or Google Colab.
    * Ensure `Dry_Bean_Dataset.xlsx` is in the same directory.
    * Run all cells to reproduce the results and plots.

## 🤝 Future Work
* Integration with **Hyperspectral Imaging** to resolve the overlap between *Sira* and *Dermason* varieties.
* Deployment of the Random Forest model on **Raspberry Pi** for real-time sorting tests.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
