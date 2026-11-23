# AI-Based Early Prediction of Diabetes Using Machine Learning

## Project Overview
This project is an AI-based diabetes risk prediction system developed using Machine Learning. It uses a Random Forest Classifier to predict whether a person has a high or low risk of diabetes based on health-related input parameters.

The application is developed using Python and Streamlit.

## Features
- User-friendly web interface.
- Diabetes risk prediction.
- Random Forest Machine Learning algorithm.
- Health parameter input form.
- High Risk / Low Risk prediction.
- Model accuracy evaluation.

## Technologies Used
- Python
- Machine Learning
- Random Forest Classifier
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Input Features
1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## Model Performance
- Algorithm: Random Forest Classifier
- Reported Accuracy: 77%

*The accuracy should be confirmed using an independent test dataset.*

## Project Structure
```text
AI-Based-Diabetes-Prediction/
│
├── app.py
├── train.py
├── diabetes.csv
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder
```bash
cd AI-Based-Diabetes-Prediction
```

### 3. Install required libraries
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning model
```bash
python train.py
```

### 5. Run the Streamlit application
```bash
streamlit run app.py
```

## Disclaimer
This project is developed for educational purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

## Author
Developed as a BCA Machine Learning Project.