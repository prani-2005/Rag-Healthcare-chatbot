import requests
r = requests.get("https://api.pinecone.io")
print(r.status_code)
