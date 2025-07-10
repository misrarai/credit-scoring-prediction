import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load dataset
df = pd.read_csv('credit_risk_dataset.csv')

# Fill missing values and preprocess
def preprocess_data(df):
    # Separate features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Identify columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', cat_imputer),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    return X, y, preprocessor

X, y, preprocessor = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit preprocessor
preprocessor.fit(X_train)
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Train models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

best_model = None
best_auc = 0
for name, model in models.items():
    model.fit(X_train_trans, y_train)
    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f'{name} ROC-AUC: {auc:.4f}')
    print(classification_report(y_test, y_pred))
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# Save best model and preprocessor
joblib.dump(best_model, 'best_credit_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
print(f'Best model: {best_model_name} with ROC-AUC: {best_auc:.4f}')
