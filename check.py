import requests
import json

url = "http://127.0.0.1:5000/predict"

# Input data to send in the POST request
data = {
    "Month": "December",
    "Season": "Winter",
    "Budget": "Medium",
    "Activity_Preference": "Adventure",
    "Group_Size": 5
}

# Send POST request with input data
response = requests.post(url, json=data)

# Print the status code and response content
print("Status Code:", response.status_code)
print("Response Content:", response.json())  # This will print the JSON response

