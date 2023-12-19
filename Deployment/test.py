import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': 'https://cdn.pixabay.com/photo/2021/01/13/14/51/airbus-a400m-atlas-5914332_960_720.jpg'}

result = requests.post(url, json=data).json()
print()
print("Prediction for aircraft",data['url'],":")
print(result)
print()