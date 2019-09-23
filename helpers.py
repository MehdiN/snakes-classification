# helper file to build a small data set using Bing Image Search APÏ
# Based on the tutorial https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset

import requests
import os
from requests import exceptions
from PIL import Image
from io import BytesIO

URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# get your API KEY from AZURE
def get_api_key(path):
    with open(path,'r') as f:
        api_key = f.readline()
        f.close
    return api_key

API_KEY = get_api_key("api_key")

# GET THE IMAGE 
def get_img(query,count=50,offset=0):
    header = {"Ocp-Apim-Subscription-Key" : API_KEY}
    params = {
            "q" : query,
            "offset" : offset,
            "imageType": "photo",
            "count": count
            }
    response = requests.get(URL,headers=header,params=params)
    response.raise_for_status()
    result = response.json()
    return result


EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])


def download_img(result,name,path,index):
    i = index
    for elem in result["value"]:
        try:
            r = requests.get(elem["contentUrl"],timeout=30,stream=True)
            if r.status_code == 200:
                ext = elem["contentUrl"]
                ext = ext[-5:len(ext)].split('.')
                ext = ext[-1]
                if ext in ['jpg','jpeg','png']:
                    with open(path +'/' + name + "_" + str(i) + '.' + ext,"wb") as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                        f.close()
                else:
                    pass
            i += 1
        except Exception as e:
            if type(e) in EXCEPTIONS:
                continue


def download_dataset():
    queries = ["Agkistrodon contortrix","Agkistrodon piscivorus","Nerodia sipedon","snakes"]
    classes = ["copperhead","cottonmouth","watersnake","others"]
    batch_size = 50
    data_set_size = 600 # per class
    for q,c in zip(queries,classes):
        result = get_img(query=q,count=batch_size,offset=0)
        max_results = min([data_set_size,result["totalEstimatedMatches"]])
        for offset in range(0,max_results,batch_size):
            result = get_img(query=q,count=batch_size,offset=offset)
            download_img(result,name=c,path="dataset/"+c,index=offset)


if __name__ == "__main__":
    download_dataset()