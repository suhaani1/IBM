import requests

url = "http://127.0.0.1:5000/predict"
data = {"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
