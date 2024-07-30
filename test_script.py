import requests

# Define the URL of the FastAPI endpoint
url = 'https://test-cicd-0b79cce14943.herokuapp.com/inference'

# Example payload with valid data
data = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 11111,
    "education": "Bachelors",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 1111,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}

# Send POST request to the FastAPI endpoint
response = requests.post(url, json=data)

# Extract the result and status code
result = response.json()
status_code = response.status_code

# Print the results
print(f"Status Code: {status_code}")
print(f"Model Inference Result: {result}")
