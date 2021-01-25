import requests
import json


url = "http://34.87.169.149/judge/3/upload_file/"
#url = "http://127.0.0.1:8000/judge/3/upload_file/"
fin = open('ex3_linreg_tf.ipynb', 'rb')
files = {'file': fin}
try:
    data_obj = {'name': input('username: ')}
    r = requests.post(url, files=files, data=data_obj)
    #print(r.text)
    res = json.loads(r.text)
    print('TOTAL SCORE: ', res['score'])
    print(res['message'])
finally:
    fin.close()
