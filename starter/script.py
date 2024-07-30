import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from eval_performance_on_slice import evaluate_performance_on_feature_slices  # Replace 'your_module' with your actual module name
from ml.data import process_data
from ml.model import (
    train_model, 
    inference,
  compute_model_metrics,
)
# Sample data

data = pd.read_csv('../data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# Prepare data
X = train[['workclass', 'education']]
y = train['salary']

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train, y_train, _, _ = process_data(
    X, categorical_features=cat_features, label="salary", training=True
)
# Train model
model = train_model(X_train, y_train)

# Evaluate performance on the feature slices
evaluate_performance_on_feature_slices(model, data, 'education', 'salary', 'slice_output.txt')
