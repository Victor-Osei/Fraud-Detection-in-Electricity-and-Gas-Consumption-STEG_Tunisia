# Fraud Detection in Electricity and Gas Consumption - STEG Tunisia

## Project Overview
This project implements a machine learning solution for detecting fraudulent behavior in electricity and gas consumption for STEG (Société Tunisienne de l'Electricité et du Gaz). The system analyzes client data and invoice patterns to identify potential cases of fraud.

## Problem Description
STEG faces challenges with fraudulent activities in electricity and gas consumption, which leads to significant revenue losses. This project aims to:
- Identify suspicious consumption patterns
- Detect potential fraud cases based on historical data
- Help STEG prioritize investigations of suspicious cases

## Data Description
The dataset consists of two main components:

### Training Data (`data/train/`)
- `client_train.csv`: Client information including:
  - Client ID
  - District
  - Client category
  - Region
  - Creation date
  - Target (fraud: 1, no fraud: 0)
- `invoice_train.csv`: Invoice details including:
  - Consumption levels (1-4)
  - Counter information
  - Invoice dates

### Test Data (`data/test/`)
- Similar structure to training data but without the target variable

## Project Structure
```
├── data/
│   ├── train/
│   │   ├── client_train.csv
│   │   └── invoice_train.csv
│   ├── test/
│   │   ├── client_test.csv
│   │   └── invoice_test.csv
│   └── SampleSubmission.csv
├── models/
│   └── lightgbm_fraud_detection.joblib
├── notebook/
│   ├── fraud_detection.ipynb
│   └── predict.py
├── requirements.txt
└── README.md
```

## Model Performance
The LightGBM model achieves:
- Accuracy: 94%
- ROC AUC Score: 0.769
- Precision for fraud detection: 38%
- Recall for fraud detection: 1%

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Fraud-Detection-in-Electricity-and-Gas-Consumption-STEG_Tunisia.git
cd Fraud-Detection-in-Electricity-and-Gas-Consumption-STEG_Tunisia
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv steg_env
source steg_env/bin/activate  # On Windows: steg_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training and Prediction
To train the model and make predictions:
```bash
python notebook/predict.py
```
This will:
- Load and preprocess the data
- Train the LightGBM model
- Save the trained model to `models/lightgbm_fraud_detection.joblib`
- Generate predictions for the test set
- Save predictions to `submission.csv`

### Using the Trained Model
To use the saved model for predictions:
```python
import joblib
model = joblib.load('models/lightgbm_fraud_detection.joblib')
predictions = model.predict_proba(new_data)[:, 1]
```

## Features Engineering
The model uses several engineered features:
- Aggregated consumption levels
- Transaction counts per client
- Account age
- Temporal patterns in consumption

## Model Details
- Algorithm: LightGBM Classifier
- Parameters:
  - boosting_type: 'gbdt'
  - num_iterations: 500
  - learning_rate: 0.05
  - num_leaves: 31

## Future Improvements
Potential areas for improvement:
1. Feature engineering:
   - Add more temporal features
   - Create consumption pattern indicators
2. Model enhancements:
   - Try different algorithms (XGBoost, CatBoost)
   - Implement ensemble methods
3. Handling class imbalance:
   - Experiment with SMOTE or other sampling techniques
   - Try different class weights

## License
This project is licensed under the GNU General Public License - see the LICENSE file for details.

## Contributor
- Victor Osei Duah

## Acknowledgments
- ZINDI for providing the dataset

