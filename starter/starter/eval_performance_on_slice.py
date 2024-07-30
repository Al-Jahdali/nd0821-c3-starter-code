import pandas as pd
from ml.data import process_data
from ml.model import (
    train_model, 
    inference,
  compute_model_metrics,
)
import joblib
from sklearn.model_selection import train_test_split

def evaluate_performance_on_feature_slices( data, categorical_features, output_file='slice_output.txt'):
    
    _, test_data = train_test_split(data, test_size=0.20, random_state=42)
    
    # Load pre-trained model and preprocessing objects
    model = joblib.load('../model/model.pkl')
    encoder = joblib.load('../model/encoder.pkl')
    label_binarizer = joblib.load('../model/lb.pkl')

    # Initialize a dictionary to store evaluation results
    results = {
        'Feature': [],
        'Category': [],
        'Precision': [],
        'Recall': [],
        'F-beta': []
    }
    print(test_data.columns)
    print(f"cat values: {categorical_features}")
    # Iterate through each categorical feature and its unique values
    for feature in categorical_features:
        # unique_values = test_data[feature].unique()
        
        for value in test_data[feature].unique():
            # Slice the data for the current feature value
            subset = test_data[test_data[feature] == value]
            
            # Process the subset data
            X_subset, y_subset, _, _ = process_data(
                subset, categorical_features=categorical_features, label='salary', training=False,
                encoder=encoder, lb=label_binarizer
            )
            
            # Make predictions on the processed data
            predictions = model.predict(X_subset)
            
            # Compute performance metrics
            precision, recall, fbeta = compute_model_metrics(y_subset, predictions)
            
            # Append the results to the dictionary
            results['Feature'].append(feature)
            results['Category'].append(value)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F-beta'].append(fbeta)
    
    # Convert the results dictionary to a DataFrame and save to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    # Define the categorical features to evaluate
    
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
    # Load the dataset
    data = pd.read_csv('../data/census.csv')
    data.columns = data.columns.str.strip()
    evaluate_performance_on_feature_slices(data, cat_features)

