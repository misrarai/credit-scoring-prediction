# Credit Scoring Prediction

This project predicts credit default risk using machine learning. It includes a Streamlit web app for user interaction.

## Features
- Data preprocessing (missing value imputation, encoding, scaling)
- Model training (Random Forest, Logistic Regression, Decision Tree)
- Model selection based on ROC-AUC
- Streamlit UI for predictions

## Files
- `train_model.py`: Preprocesses data, trains models, saves the best model and preprocessor.
- `app.py`: Streamlit app for user input and prediction.
- `credit_risk_dataset.csv`: Dataset for training.
- `requirements.txt`: Python dependencies.

## Usage
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train and save the model:
   ```
   python train_model.py
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Model Evaluation
- Models are evaluated using ROC-AUC, Precision, Recall, and F1-Score.
- The best model is saved as `best_credit_model.joblib` and the preprocessor as `preprocessor.joblib`.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## Author
- Your Name
