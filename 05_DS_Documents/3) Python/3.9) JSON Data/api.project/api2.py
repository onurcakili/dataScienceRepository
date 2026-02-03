from pip._vendor import requests
import json

url = "https://jsonplaceholder.typicode.com/posts"  # URL yapıştır

payload = {}
headers= {}

response = requests.request("GET", url, headers=headers, data = payload)

result = json.loads(response.text)


print(result)

