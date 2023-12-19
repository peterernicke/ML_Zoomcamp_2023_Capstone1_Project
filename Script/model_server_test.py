import requests

url='http://localhost:9696/predict'

data = {
    #"url": "https://cdn.pixabay.com/photo/2021/01/13/14/51/airbus-a400m-atlas-5914332_960_720.jpg"
    "url": "https://imgr1.flugrevue.de/Tupolew-Tu-160-benannt-nach-dem-Chefkonstrukteur-Valentin-Iwanowitsch-Bliznjuk-article169Gallery-1063deed-1659300.jpg"
}

response = requests.post(url, json=data).json()
print("Prediction for aircraft",data,":")
#print(data,"\n")
print(response)