# train_model_v3.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

# config
CSV_PATH = r"C:\Users\hp\OneDrive\Documents\Keepwise\Employee_Attrition_Major_Project_Final_Complete\WA_Fn-UseC_-HR-Employee-Attrition.csv"  
OUT_MODEL = "models/attrition_pipeline_v3.joblib"
os.makedirs("models", exist_ok=True)

# 1) load dataset
df = pd.read_csv(CSV_PATH)

# 2) features & target
features = [
    'Age','BusinessTravel','Department','MonthlyIncome','OverTime',
    'JobRole','JobSatisfaction','TotalWorkingYears','YearsAtCompany',
    'EnvironmentSatisfaction','WorkLifeBalance','PerformanceRating'
]
target_col = 'Attrition'  # adapt if different

# quick sanity: drop rows with missing required columns
df = df.dropna(subset=features + [target_col])

X = df[features].copy()
y = df[target_col].map({'Yes':1, 'No':0})  # adapt mapping if needed

# 3) define columns
numeric_cols = ['Age','MonthlyIncome','TotalWorkingYears','YearsAtCompany',
                'JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','PerformanceRating']
categorical_cols = ['BusinessTravel','Department','OverTime','JobRole']

# 4) pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# 5) train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)

# evaluation
print("Train score:", clf.score(X_train, y_train))
print("Test score:", clf.score(X_test, y_test))

# save
joblib.dump(clf, OUT_MODEL)
print(f"Saved pipeline to {OUT_MODEL}")
