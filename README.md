# 🎓 Student Mathematical Performance Classification

This project contains a Streamlit web application that predicts a student's mathematical performance based on their answers to a 14-question survey. The prediction is made by a pre-trained Support Vector Machine (SVM) model.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
-   [Python 3.8+](https://www.python.org/downloads/)
-   `pip` (Python package installer)

## 🛠️ Setup and Installation Guide

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine using Git:
```bash
git clone https://github.com/prathviraj-03/svm_model_mathematics.git
cd svm_model_mathematics
```
If you do not have Git, you can download the project as a ZIP file and extract it.

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep the project's dependencies isolated from your global Python environment.

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
After activation, you will see `(venv)` at the beginning of your terminal prompt.

### 3. Install Required Dependencies

Install all the necessary Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
This will install Streamlit, scikit-learn, NumPy, and other required packages.

## 🚀 Running the Application

Once the setup is complete, you can run the Streamlit web application.

1.  Make sure you are in the project's root directory (where `main.py` is located).
2.  Run the following command in your terminal:
    ```bash
    streamlit run main.py
    ```
3.  Streamlit will start a local server, and the application will automatically open in your default web browser. If it doesn't, your terminal will display a local URL (usually `http://localhost:8501`) that you can open manually.

You can now interact with the "Math Performance Predictor" by answering the questions and clicking the "Predict Performance" button.

## 📂 Project Components

-   `main.py`: The main Python script for the Streamlit web application.
-   `Student_Mathematical_Performance_Classification.ipynb`: A Jupyter Notebook detailing the data analysis, model training, and evaluation process. You can explore this to understand how the model was built.
-   `ASSETS/`: This directory contains the pre-trained machine learning assets:
    -   `svm_best_model.joblib`: The trained SVM classifier.
    -   `scaler.joblib`: The scaler used to preprocess the input data.
-   `requirements.txt`: A list of all Python dependencies required for the project.
