import requests

listing = {
    "neighbourhood": "Midtown",
    "room_type": "Private room",
    "availability_365": 100,
}

url = "http://localhost:9696/04-deploy"
response = requests.post(url, json=listing)
print(response.json())
