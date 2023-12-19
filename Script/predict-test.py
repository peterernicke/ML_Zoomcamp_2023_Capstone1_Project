import requests

url='http://localhost:9797/predict'

data = {
    "url": "https://cdn.pixabay.com/photo/2021/01/13/14/51/airbus-a400m-atlas-5914332_960_720.jpg"
}

response = requests.post(url, json=data).json()
print("Prediction for aircraft",data,":")
#print(data,"\n")
print(response)