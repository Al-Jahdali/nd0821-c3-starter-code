# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Applied the Logistic Regression classifier for predictions using the default configuration for training.
## Intended Use
This model is designed to predict a person's salary category based on their financial attributes.


## Training Data
80% of  https://archive.ics.uci.edu/ml/datasets/census+income been selected for training 
## Evaluation Data
20% of  https://archive.ics.uci.edu/ml/datasets/census+income been selected for testing 
## Metrics

Precision: 0.71
Recall: 0.61
F-beta: 0.66
## Ethical Considerations
To address ethical considerations, metrics were also calculated on data slices. This helps identify potential discrimination in the model
## Caveats and Recommendations
The data exhibits gender bias and has data imbalances that require further investigation.