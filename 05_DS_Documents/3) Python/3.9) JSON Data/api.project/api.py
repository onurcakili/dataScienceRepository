import json

with open("users.json") as users:
    data = json.load(users)
    print(data[0])
    
    