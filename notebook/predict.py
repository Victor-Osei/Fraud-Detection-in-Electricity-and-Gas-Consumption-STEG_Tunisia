import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# Define paths
base_dir = os.path.abspath(os.getcwd())
test_dir = os.path.join(base_dir, "data", "test")

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

# Load test data
print("Loading test data...")
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

# Initialize model with same parameters
print("Initializing model...")
model = LGBMClassifier(
    boosting_type='gbdt',
    num_iterations=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# Make predictions
print("Making predictions...")
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