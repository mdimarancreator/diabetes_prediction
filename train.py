
import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load the Dataset
# --------------------------------------------------

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "diabetes.csv")

# Check whether the dataset file exists
if not os.path.exists(csv_path):
    sys.exit(
        "Error: diabetes.csv was not found. "
        "Please place it in the same directory as train.py."
    )

# Read the dataset
df = pd.read_csv(csv_path)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()


# --------------------------------------------------
# 2. Validate Required Dataset Columns
# --------------------------------------------------

required_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    sys.exit(
        f"Error: The following required columns are missing: "
        f"{missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# --------------------------------------------------
# 3. Handle Missing or Invalid Zero Values
# --------------------------------------------------

# In these medical measurements, zero may represent a missing value.
# Replace such values with the median of the available non-zero values.

columns_to_impute = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_to_impute:

    median_value = df.loc[df[column] != 0, column].median()

    if pd.isna(median_value):
        sys.exit(
            f"Error: Cannot calculate a valid median for {column}."
        )

    df[column] = df[column].replace(0, median_value)


# --------------------------------------------------
# 4. Separate Features and Target Variable
# --------------------------------------------------

# Features contain the input variables.
# Outcome is the target variable to be predicted.

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# --------------------------------------------------
# 5. Split the Dataset into Training and Testing Sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Initialize and Train the Random Forest Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Evaluate Model Performance
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. Save the Trained Model
# --------------------------------------------------

model_path = os.path.join(base_dir, "model.pkl")

joblib.dump(model, model_path)

print("Model training completed successfully.")
print(f"Trained model saved at: {model_path}")
