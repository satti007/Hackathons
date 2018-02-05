import json
import requests

file = open('output.json', 'r') 

data = json.load(file)

verify_url = 'https://us-central1-lynkhacksmock.cloudfunctions.net/verifyface'

log_response = open('response.log', 'w')

for elem in data:
	r = requests.post(verify_url, elem)
	print r.text
	log_response.write(r.text + '\n')

log_response.close()