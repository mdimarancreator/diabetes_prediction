import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data (file isi folder mein honi chahiye jahan train.py hai)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "diabetes.csv")

if not os.path.exists(csv_path):
    sys.exit("Error: diabetes.csv nahi mili. Use train.py wale folder mein rakhiye.")

df = pd.read_csv(csv_path)

# Column ke naam se extra space hata do
df.columns = df.columns.str.strip()

# Zaruri columns check karo
required = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
missing = [c for c in required if c not in df.columns]
if missing:
    sys.exit(f"Error: ye columns file mein nahi mile: {missing}\nMilne wale columns: {list(df.columns)}")

# 2. Fix zero values (they mean "missing")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in cols:
    median_value = df.loc[df[c] != 0, c].median()  # zero ko chhod kar median
    df[c] = df[c].replace(0, median_value)

# 3. Split into input (X) and answer (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

# 6. Save (isi folder mein model.pkl banega)
joblib.dump(model, os.path.join(base_dir, "model.pkl"))
print("model.pkl save ho gayi.")