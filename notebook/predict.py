import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import joblib  # Add joblib import

# Define paths
base_dir = os.path.abspath(os.getcwd())
train_dir = os.path.join(base_dir, "data", "train")
test_dir = os.path.join(base_dir, "data", "test")
models_dir = os.path.join(base_dir, "models")  # Add models directory

# Create models directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Function to aggregate invoice data
def aggregate_by_client_id(invoice_data):
    aggs = {}
    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='transactions_count'))
    return pd.merge(df, agg_trans, on='client_id', how='left')

# Load training data
print("Loading training data...")
client_train = pd.read_csv(os.path.join(train_dir, "client_train.csv"))
invoice_train = pd.read_csv(os.path.join(train_dir, "invoice_train.csv"))

# Prepare training data
print("Preparing training data...")
# Aggregate invoice data
agg_train = aggregate_by_client_id(invoice_train)

# Merge with client data
train_data = pd.merge(client_train, agg_train, on='client_id', how='left')

# Process creation_date
train_data['creation_date'] = pd.to_datetime(train_data['creation_date'], format='%d/%m/%Y')
train_data['account_age_days'] = (pd.Timestamp.now() - train_data['creation_date']).dt.days

# Prepare features and target
X = train_data.drop(['client_id', 'creation_date', 'target'], axis=1)
y = train_data['target']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
print("Training model...")
model = LGBMClassifier(
    boosting_type='gbdt',
    num_iterations=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import classification_report, roc_auc_score
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]
print("\nValidation Results:")
print(classification_report(y_val, val_preds))
print("ROC AUC Score:", roc_auc_score(y_val, val_probs))

# After model training and evaluation
print("Saving the model...")
model_path = os.path.join(models_dir, 'lightgbm_fraud_detection.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Load test data
print("\nLoading test data...")
client_test = pd.read_csv(os.path.join(test_dir, "client_test.csv"))
invoice_test = pd.read_csv(os.path.join(test_dir, "invoice_test.csv"))

# Prepare test data
print("Preparing test data...")
# Aggregate invoice data
agg_test = aggregate_by_client_id(invoice_test)

# Merge with client data
test_data = pd.merge(client_test, agg_test, on='client_id', how='left')

# Process creation_date
test_data['creation_date'] = pd.to_datetime(test_data['creation_date'], format='%d/%m/%Y')
test_data['account_age_days'] = (pd.Timestamp.now() - test_data['creation_date']).dt.days

# Drop unnecessary columns
test_features = test_data.drop(['client_id', 'creation_date'], axis=1)

# One-hot encode categorical variables
test_features = pd.get_dummies(test_features, drop_first=True)

# Ensure test features have same columns as training features
missing_cols = set(X.columns) - set(test_features.columns)
for col in missing_cols:
    test_features[col] = 0
test_features = test_features[X.columns]

# Make predictions
print("Making predictions on test data...")
predictions = model.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'client_id': client_test['client_id'],
    'target': predictions
})

# Save predictions
submission_path = os.path.join(base_dir, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f"\nPredictions saved to {submission_path}")
print("\nFirst few predictions:")
print(submission.head()) 