#!/usr/bin/env python
# coding: utf-8

# ## Data Understanding

# Import Libraries

# In[ ]:


import os.path
from os import path


import zipfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


import warnings
warnings.simplefilter('ignore')


# Extract zipfiles

# In[2]:


# # Paths
# train_zip_path = "data/train/train.zip"
# test_zip_path = "data/test/test.zip"
# train_extract_path = "data/train/"
# test_extract_path = "data/test/"

# # Extract train.zip
# with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
#     zip_ref.extractall(train_extract_path)

# # Extract test.zip
# with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
#     zip_ref.extractall(test_extract_path)




# Go up one level from notebook/ to project root
base_dir = os.path.abspath(os.path.join(os.getcwd()))

# Define correct paths
train_zip_path = os.path.join(base_dir, "data", "train", "train.zip")
test_zip_path = os.path.join(base_dir, "data", "test", "test.zip")
train_extract_path = os.path.join(base_dir, "data", "train")
test_extract_path = os.path.join(base_dir, "data", "test")

print("Current directory:", os.getcwd())
print("Base directory:", base_dir)
print("Train zip path:", train_zip_path)

# Extract files if needed
if not os.path.exists(os.path.join(train_extract_path, "client_train.csv")):
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(train_extract_path)

if not os.path.exists(os.path.join(test_extract_path, "client_test.csv")):
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(test_extract_path)

print("Extraction completed.")


# Read datasets

# In[3]:


# # Train sets
# client_train = pd.read_csv("data/train/client_train.csv")
# invoice_train = pd.read_csv("data/train/invoice_train.csv")

# # Test sets
# client_test = pd.read_csv("data/test/client_test.csv")
# invoice_test = pd.read_csv("data/test/invoice_test.csv")

# # Sample Submission
# sample_submission = pd.read_csv("data/SampleSubmission.csv")


# Train sets
client_train = pd.read_csv(os.path.join(train_extract_path, "client_train.csv"))
invoice_train = pd.read_csv(os.path.join(train_extract_path, "invoice_train.csv"))

# Test sets
client_test = pd.read_csv(os.path.join(test_extract_path, "client_test.csv"))
invoice_test = pd.read_csv(os.path.join(test_extract_path, "invoice_test.csv"))

# Sample Submission
sample_submission = pd.read_csv(os.path.join(base_dir, "data", "SampleSubmission.csv"))


# #### Exploratory Data Analysis

# In[4]:


client_train.head()


# In[5]:


invoice_train.head()


# In[6]:


# compare size of the various datasets
print(client_train.shape, invoice_train.shape, client_test.shape, invoice_train.shape)


# In[7]:


# Get a summary for all numerical columns
invoice_train.describe()


# In[8]:


client_train.describe()


# In[9]:


client_train.info()


# In[10]:


invoice_train.info()


# In[11]:


# check unique values on the client train data

for col in client_train.columns:
    print(f"{col} - {client_train[col].nunique()}")


# In[12]:


# check unique values on the invoice train data

for col in invoice_train.columns:
    print(f"{col} - {invoice_train[col].nunique()}")


# In[13]:


#check for missing values
invoice_train.isnull().sum()


# In[14]:


#check for missing values
client_train.isnull().sum()


# Observation: Luckily no missing values in Train datasets (client_train and invoice_train)

# In[ ]:





# Fraud Distribution

# In[15]:


client_train['target'].value_counts().plot(kind='bar', title='Fraudulent vs Non-Fraudulent Clients')
plt.xticks(ticks=[0,1], labels=['Non-Fraud', 'Fraud'])
plt.ylabel('Number of Clients')
plt.show()


# In[16]:


#Visualize fraudulent activities
fraudactivities = client_train.groupby(['target'])['client_id'].count()
plt.bar(x=fraudactivities.index, height=fraudactivities.values, tick_label = [0,1])
plt.title('Fraud - Target Distribution')
plt.show()


# Region and District Fraud Distribution

# In[17]:


#Visualize client distribution across districts and regions
for col in ['disrict','region']:
    region = client_train.groupby([col])['client_id'].count()
    plt.bar(x=region.index, height=region.values)
    plt.title(col+' distribution')
    plt.show()


# ### Feature Engineering

# Convert invoice_date to datetime

# In[18]:


# this is done on both the invoice train and invoice test
for df in [invoice_train,invoice_test]:
    df['invoice_date'] = pd.to_datetime(df['invoice_date']) 


# Label Encoding counter_type (ELEC & GAZ)

# In[19]:


#encode labels in categorical column
d={"ELEC":0,"GAZ":1}
invoice_train['counter_type']=invoice_train['counter_type'].map(d)
invoice_test['counter_type']=invoice_test['counter_type'].map(d)


# Convert Categorical Columns

# In[20]:


# convert categorical columns to int for model
client_train['client_catg'] = client_train['client_catg'].astype(int)
client_train['disrict'] = client_train['disrict'].astype(int)

client_test['client_catg'] = client_test['client_catg'].astype(int)
client_test['disrict'] = client_test['disrict'].astype(int)


# Invoice Aggregation by Client

# In[39]:


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
            .reset_index(name='{}transactions_count'.format('1')))
    return pd.merge(df, agg_trans, on='client_id', how='left')


# In[22]:


# group invoice data by client_id
agg_train = aggregate_by_client_id(invoice_train)


# In[23]:


print(agg_train.shape)
agg_train.head()


# In[24]:


#merge aggregate data with client dataset
train_data = pd.merge(client_train,agg_train, on='client_id', how='left')


# In[25]:


train_data.head()


# In[26]:


train_data.shape


# ### Modeling

# In[27]:


# select features and target variable
# Drop identifiers and target
X = train_data.drop(['client_id', 'target'], axis=1)
y = train_data['target']


# One-Hot Encoding for Categorical Variables

# In[28]:


X = pd.get_dummies(X, drop_first=True)


# Check for Missing Values

# In[29]:


X.isnull().sum().sort_values(ascending=False)


# Split Data (Train-Test)

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# #### Model Selection and Training

# RF_Model

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_probs = rf_model.predict_proba(X_val)[:,1]

print("Random Forest Classification Report:\n", classification_report(y_val, rf_preds))
print("ROC AUC Score:", roc_auc_score(y_val, rf_probs))


# LightGBM 

# In[33]:


from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Initialize LightGBM model
lgbm_model = LGBMClassifier(boosting_type='gbdt', num_iterations=500, random_state=42)

# Train the model
lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set
lgbm_preds = lgbm_model.predict(X_val)
lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

# Print classification report and AUC score
print("LightGBM Classification Report:\n", classification_report(y_val, lgbm_preds))
print("ROC AUC Score:", roc_auc_score(y_val, lgbm_probs))


# 

# **Summary of Model Performance**

# In[35]:


from tabulate import tabulate
import pandas as pd

# Define the data
performance_data = {
    "Metric": ["Accuracy", "Precision (1.0)", "Recall (1.0)", "F1-Score (1.0)", "ROC AUC Score"],
    "Random Forest": ["94%", "0.27", "0.01", "0.03", "0.707"],
    "LightGBM": ["94%", "0.40 ✅", "0.02 ✅", "0.03", "0.750 ✅"]
}

# Create DataFrame
performance_df = pd.DataFrame(performance_data)

# Pretty print using tabulate
print(tabulate(performance_df, headers='keys', tablefmt='fancy_grid', showindex=False))


# **Interpretation:**
# 
# Both models perform equally well on predicting the majority class (0.0).
# 
# LightGBM performs better at identifying the minority class (1.0) — even though the recall is still low, it's double what the Random Forest gives.
# 
# The ROC AUC score is significantly higher for LightGBM, which tells us that it's better at ranking positive vs. negative examples overall.
# 
# **Conclusion:**
# 
# Use LightGBM for final model selection and test set predictions. It's showing better recall and higher AUC, which are more important in imbalanced classification problems like this.

# 

# **Model Selection and Prediction on Test dataset**

# In[48]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import lightgbm as lgb
import numpy as np
from scipy.sparse import hstack

# 1. Aggregate test invoice data
agg_invoice_test = aggregate_by_client_id(invoice_test)

# 2. Merge with client_test
test_data = pd.merge(client_test, agg_invoice_test, on='client_id', how='left')

# 3. Identify and encode high-cardinality columns
high_card_cols = [col for col in test_data.select_dtypes(include='object') if test_data[col].nunique() > 100]
le = LabelEncoder()
for col in high_card_cols:
    test_data[col] = le.fit_transform(test_data[col].astype(str))

# 4. Identify low-cardinality columns
low_card_cols = [col for col in test_data.select_dtypes(include='object') if col not in high_card_cols]
print("Low Cardinality Columns:", low_card_cols)

# 5. Check if low-card-cols contain valid data
print(test_data[low_card_cols].nunique())  # Show number of unique values in low-card columns
print(test_data[low_card_cols].isnull().sum())  # Show number of missing values in low-card columns

# 6. Handle missing values by filling with a placeholder (e.g., 'Missing')
test_data[low_card_cols] = test_data[low_card_cols].fillna('Missing')

# 7. Verify data in low-cardinality columns
print(test_data[low_card_cols].head())  # Display the first few rows to ensure data is filled

# 8. Convert low-cardinality columns to dictionary format
dict_data = test_data[low_card_cols].astype(str).to_dict(orient='records')
print(f"First 5 records in dict_data: {dict_data[:5]}")  # Check the first 5 records

# 9. One-hot encode low-cardinality columns using DictVectorizer (sparse matrix)
if dict_data:  # Check if dict_data is not empty
    vec = DictVectorizer(sparse=True)
    X_sparse = vec.fit_transform(dict_data)

    # 10. Drop low-cardinality columns and combine numeric columns with sparse matrix
    test_data.drop(columns=low_card_cols, inplace=True)
    X_dense = test_data.select_dtypes(include=[np.number])

    # 11. Combine sparse and dense features into one matrix
    X_final = hstack([X_sparse, X_dense])

    # 12. Predict using LightGBM with the final feature matrix (sparse + numeric)
    test_preds = lgbm_model.predict_proba(X_final)[:, 1]
else:
    print("No valid data for low-cardinality columns.")


# In[49]:


# 3. Identify high-cardinality columns
high_card_cols = [col for col in test_data.select_dtypes(include='object') if test_data[col].nunique() > 100]

# 4. Identify low-cardinality columns with a smaller threshold (e.g., <= 10 unique values)
low_card_cols = [col for col in test_data.select_dtypes(include='object') if col not in high_card_cols and test_data[col].nunique() <= 10]

print("Low Cardinality Columns:", low_card_cols)

# 5. Check if low-card-cols contain valid data
print(test_data[low_card_cols].nunique())  # Show number of unique values in low-card columns
print(test_data[low_card_cols].isnull().sum())  # Show number of missing values in low-card columns

# 6. Handle missing values by filling with a placeholder (e.g., 'Missing')
test_data[low_card_cols] = test_data[low_card_cols].fillna('Missing')

# 7. Verify data in low-cardinality columns
print(test_data[low_card_cols].head())  # Display the first few rows to ensure data is filled

# 8. Convert low-cardinality columns to dictionary format
dict_data = test_data[low_card_cols].astype(str).to_dict(orient='records')
print(f"First 5 records in dict_data: {dict_data[:5]}")  # Check the first 5 records

# 9. One-hot encode low-cardinality columns using DictVectorizer (sparse matrix)
if dict_data:  # Check if dict_data is not empty
    vec = DictVectorizer(sparse=True)
    X_sparse = vec.fit_transform(dict_data)

    # 10. Drop low-cardinality columns and combine numeric columns with sparse matrix
    test_data.drop(columns=low_card_cols, inplace=True)
    X_dense = test_data.select_dtypes(include=[np.number])

    # 11. Combine sparse and dense features into one matrix
    X_final = hstack([X_sparse, X_dense])

    # 12. Predict using LightGBM with the final feature matrix (sparse + numeric)
    test_preds = lgbm_model.predict_proba(X_final)[:, 1]
else:
    print("No valid data for low-cardinality columns.")


# In[50]:


# 1. Print the unique values count for each categorical column
categorical_cols = test_data.select_dtypes(include='object').columns
print(f"Categorical columns in the dataset: {categorical_cols}")
for col in categorical_cols:
    print(f"{col}: {test_data[col].nunique()} unique values")

# 2. Identify low-cardinality columns using a lower threshold (e.g., <= 10 unique values)
low_card_cols = [col for col in categorical_cols if test_data[col].nunique() <= 50]
print("Low Cardinality Columns:", low_card_cols)

# 3. Check the number of unique values and missing values for low-cardinality columns
print(test_data[low_card_cols].nunique())  # Show number of unique values in low-card columns
print(test_data[low_card_cols].isnull().sum())  # Show number of missing values in low-card columns

# 4. Fill missing values with 'Missing'
test_data[low_card_cols] = test_data[low_card_cols].fillna('Missing')

# 5. Check first few rows after filling missing values
print(test_data[low_card_cols].head())

# 6. Convert low-cardinality columns to dictionary format
dict_data = test_data[low_card_cols].astype(str).to_dict(orient='records')
print(f"First 5 records in dict_data: {dict_data[:5]}")  # Check the first 5 records

# 7. One-hot encode low-cardinality columns using DictVectorizer (sparse matrix)
if dict_data:  # Check if dict_data is not empty
    vec = DictVectorizer(sparse=True)
    X_sparse = vec.fit_transform(dict_data)

    # 8. Drop low-cardinality columns and combine numeric columns with sparse matrix
    test_data.drop(columns=low_card_cols, inplace=True)
    X_dense = test_data.select_dtypes(include=[np.number])

    # 9. Combine sparse and dense features into one matrix
    X_final = hstack([X_sparse, X_dense])

    # 10. Predict using LightGBM with the final feature matrix (sparse + numeric)
    test_preds = lgbm_model.predict_proba(X_final)[:, 1]
else:
    print("No valid data for low-cardinality columns.")


# In[52]:


# # 1. Check the data types of all columns
print(test_data.dtypes)

# # 2. Manually define categorical columns if necessary
# categorical_cols = ['client_id', 'product_type', 'other_column']  # Replace with actual columns
# print(f"Manually defined categorical columns: {categorical_cols}")

# # 3. Convert relevant columns to categorical if they're not correctly recognized
# test_data['client_id'] = test_data['client_id'].astype(str)
# test_data['product_type'] = test_data['product_type'].astype(str)

# # 4. Identify low-cardinality columns using a threshold
# low_card_cols = [col for col in categorical_cols if test_data[col].nunique() <= 10]
# print("Low Cardinality Columns:", low_card_cols)

# # 5. Check the number of unique values and missing values in low-cardinality columns
# print(test_data[low_card_cols].nunique())  # Show number of unique values in low-card columns
# print(test_data[low_card_cols].isnull().sum())  # Show number of missing values in low-card columns

# # 6. Fill missing values with 'Missing' in low-cardinality columns
# test_data[low_card_cols] = test_data[low_card_cols].fillna('Missing')

# # 7. Convert low-cardinality columns to dictionary format
# dict_data = test_data[low_card_cols].astype(str).to_dict(orient='records')
# print(f"First 5 records in dict_data: {dict_data[:5]}")

# # 8. One-hot encode low-cardinality columns using DictVectorizer (sparse matrix)
# if dict_data:  # Check if dict_data is not empty
#     vec = DictVectorizer(sparse=True)
#     X_sparse = vec.fit_transform(dict_data)

#     # 9. Drop low-cardinality columns and combine numeric columns with sparse matrix
#     test_data.drop(columns=low_card_cols, inplace=True)
#     X_dense = test_data.select_dtypes(include=[np.number])

#     # 10. Combine sparse and dense features into one matrix
#     from scipy.sparse import hstack
#     X_final = hstack([X_sparse, X_dense])

#     # 11. Predict using LightGBM with the final feature matrix (sparse + numeric)
#     test_preds = lgbm_model.predict_proba(X_final)[:, 1]
# else:
#     print("No valid data for low-cardinality columns.")


# In[58]:


print(test_data.columns.tolist())


# In[ ]:





# In[62]:


from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# 1. Define categorical columns used during training
cat_cols = vec.feature_names_  # Safely use fitted vectorizer's feature names if possible

# 2. Make sure to select the original categorical columns used during training
# If they were 'region' and 'creation_date', include them manually if needed:
cat_features = ['region', 'creation_date']

# 3. Cast to string and fill NAs
for col in cat_features:
    if col in test_data.columns:
        test_data[col] = test_data[col].astype(str).fillna("Missing")
    else:
        # Add missing column with 'Missing' if it's not in test data (for safety)
        test_data[col] = "Missing"

# 4. Apply DictVectorizer transform
dict_data = test_data[cat_features].to_dict(orient='records')
X_sparse = vec.transform(dict_data)  # Must use same vec as in training

# 5. Get numerical features (used in training)
numeric_cols = [col for col in test_data.columns if col not in cat_features]
X_dense = test_data[numeric_cols]

# 6. Combine
X_final = hstack([X_sparse, X_dense])

# 7. Predict
assert hasattr(lgbm_model, 'booster_'), "Model is not fitted yet."
test_preds = lgbm_model.predict_proba(X_final)[:, 1]


# In[ ]:





# **Submission File**

# In[ ]:


# Prepare submission file
submission = pd.DataFrame({
    "client_id": client_test["client_id"],
    "target": test_preds
})

# Export submission file
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

