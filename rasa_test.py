import requests
def sendtorasa(response):
    res = {
        "sender" : "test",
        "message" : response
    }       
    print("Sending Response to rasa")
    rasa_output = requests.post(url="http://172.16.15.216:5005/webhooks/rest/webhook", json=res).json()
    print(f'response from rasa {rasa_output}')
    return rasa_output[0]['text'], rasa_output[1]['custom']['action']


sendtorasa("which headsets have the best resolution")