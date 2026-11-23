# 🩺 AI-Based Early Prediction of Diabetes Using Machine Learning

\<p align="center">   \<b>An intelligent diabetes risk prediction application powered by Random Forest Machine Learning.\</b> \</p>

## 📌 About the Project

The AI-Based Early Prediction of Diabetes is a machine learning project designed to estimate an individual's diabetes risk based on health-related parameters.

The system uses a Random Forest Classifier to analyze input features and generate a High Risk or Low Risk prediction through an interactive Streamlit web application.

This project demonstrates how machine learning can be applied to healthcare-related risk assessment.

## ✨ Key Features

- Interactive and user-friendly Streamlit interface.
- Diabetes risk prediction using Random Forest.
- Eight health-related input parameters.
- High Risk / Low Risk classification.
- Data preprocessing and model training.
- Machine learning model saved using Joblib.

## 🛠️ Technologies Used

| Technology    | Purpose                  |
| ------------- | ------------------------ |
| Python        | Programming language     |
| Streamlit     | Web application          |
| Pandas        | Data processing          |
| NumPy         | Numerical operations     |
| Scikit-learn  | Machine learning         |
| Random Forest | Classification algorithm |
| Joblib        | Model serialization      |

## 📊 Input Features

The prediction system uses the following parameters:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

**Reported Model Accuracy: 77%**

Note: This is the currently reported accuracy. It should be verified using an independent test dataset before being treated as the final model performance.

## 📁 Project Structure

```text
diabetes_prediction/
│
├── app.py
├── train.py
├── diabetes.csv
├── model.pkl
├── README.md
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mdimarancreator/diabetes_prediction.git
```

### 2. Navigate to Project Directory

```bash
cd diabetes_prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Model

```bash
python train.py
```

### 7. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔄 Project Workflow

1. Collect diabetes dataset.
2. Load and preprocess the dataset.
3. Separate input features and target variable.
4. Split the dataset into training and testing sets.
5. Train the Random Forest Classifier.
6. Evaluate model performance.
7. Save the trained model.
8. Load the model in Streamlit.
9. Accept user health parameters.
10. Display the predicted risk category.

## 🎯 Project Objectives

- Apply machine learning to healthcare-related data.
- Understand Random Forest classification.
- Develop an interactive prediction application.
- Practice data preprocessing and model evaluation.
- Build a practical end-to-end machine learning project.

## 🔮 Future Enhancements

- Add additional machine learning algorithms.
- Improve model performance through validation and tuning.
- Add confusion matrix and ROC-AUC visualization.
- Display prediction probabilities.
- Improve dashboard design.
- Generate downloadable prediction reports.

## ⚠️ Disclaimer

This application is developed for educational and academic purposes only. Its predictions are not medical diagnoses and must not replace professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

**MD IMARAN**

BCA Machine Learning Project

GitHub: [https://github.com/mdimarancreator](https://github.com/mdimarancreator)

---

⭐ If you find this project interesting, consider giving the repository a star!
