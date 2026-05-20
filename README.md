# Automated ECG Heartbeat Classification Pipeline for Arrhythmia Detection

This repository contains an end-to-end Machine Learning and Big Data Engineering pipeline designed for the automated preprocessing, analysis, and multi-class classification of Electrocardiogram (ECG) signals using the MIT-BIH Arrhythmia Dataset. The system architecture leverages an Ensemble Learning framework optimized for high-dimensional clinical telemetry, cost-sensitive learning to counter class imbalance, and a production-ready API/GUI deployment matrix.

---

## Technical Architecture Overview

The engineering layout is structured into a modular pipeline ensuring repeatable validation and predictable throughput:

1. Data Ingestion & Imputation: Resolves high-volume tabular feature arrays and handles missing vectors using contextual row-wise column mean imputation.
2. High-Dimensional Variance Filtering: Employs an unsupervised VarianceThreshold filter to eliminate zero-variance and non-informative static attributes.
3. Linear MinMax Amplitude Scaling: Normalizes signal variations to prevent numerical dominance across the multi-stage decision trees.
4. Stratified Data Split: Uses sample-proportional partitioning to preserve minority-class boundaries during the validation matrix split.
5. Cost-Sensitive Ensemble Training: Implements a highly parallelized Random Forest Classifier with balanced penalty matrices to mitigate severe class skewness.
6. Explainable AI (XAI) Matrix: Extracts structural feature importance rankings to decode the neural markers utilized during real-time diagnostic decisions.

---

## Dataset Acquisition

Due to institutional data size constraints and standard software engineering practices regarding Version Control Systems, the raw, heavy dataset files are managed outside the repository footprint. 

You can download the raw data directly from the verified mirror below:
- Download Link: [Google Drive Hosted Datasets](https://drive.google.com/drive/folders/YOUR_SHARED_DRIVE_FOLDER_LINK_HERE)

### File Structural Specifications:
- mitbih_train.csv: 392.44 MB (Used for pipeline fitting, variance tracking, and model optimization)
- mitbih_test.csv: 98.13 MB (Preserved exclusively for out-of-sample generalization testing)

---

## Installation and Environment Setup

To deploy the workspace locally, ensure you have a clean Python environment configured. Follow the execution blocks below:

### 1. Clone the Workspace
```bash
git clone [https://github.com/AHMED-HISHAM-DEV/ECG-Heartbeat-Classification-BigData.git](https://github.com/AHMED-HISHAM-DEV/ECG-Heartbeat-Classification-BigData.git)
cd ECG-Heartbeat-Classification-BigData
```
### 2. Configure Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate


### 3. Install Production Dependencies
pip install -r requirements.txt
Dependencies include: scikit-learn, joblib, pandas, numpy, matplotlib, seaborn, flask/fastapi.

Pipeline Execution
To run the complete model execution pipeline, execute the core script.
