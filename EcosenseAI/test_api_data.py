import requests

url = "https://65e4-117-239-78-56.ngrok-free.app/arduino-data"

try:
    response = requests.get(url)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)

