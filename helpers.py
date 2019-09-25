# helper file to build a small data set using Bing Image Search APÏ
# Based on the tutorial https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset

import requests
import os
from requests import exceptions
from PIL import Image
from io import BytesIO
import numpy as np

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


def download_dataset(path_to_dataset):
    queries = ["Agkistrodon contortrix","Agkistrodon piscivorus","Nerodia sipedon","snakes"]
    classes = ["copperhead","cottonmouth","watersnake","others"]
    batch_size = 50
    data_set_size = 600 # per class
    for q,c in zip(queries,classes):
        result = get_img(query=q,count=batch_size,offset=0)
        max_results = min([data_set_size,result["totalEstimatedMatches"]])
        for offset in range(0,max_results,batch_size):
            result = get_img(query=q,count=batch_size,offset=offset)
            download_img(result,name=c,path=path_to_dataset+c,index=offset)


def rename(name,path_dir,path_dest):
    i = 0;
    for filename in os.listdir(path_dir):
        dst = name + "_" + str(i) + ".jpg"
        src = path_dir + filename
        dst = path_dest + dst
        os.rename(src,dst)
        i += 1


## SIMPLE IMAGES TRANSFORMATIONS


def flip_img(src_path,dest_path):
    i = 0
    for filename in os.listdir(src_path):
        if os.path.isfile(src_path+filename):
            img = Image.open(src_path+filename)
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_flip.save(dest_path+filename+'_'+"flip"+'_'+str(i))
            i += 1


def rotate_img(src_path,dest_path):
    i = 0
    for filename in os.listdir(src_path):
        if os.path.isfile(src_path+filename):
            img = Image.open(src_path+filename)
            dim = np.size(img)
            # don't rotate small images
            if dim[0] >= 256 and dim[1] >= 256:
                img_rot_20 = Image.Image.rotate(img,20)
                img_rot_30 = Image.Image.rotate(img,30)
                img_rot_20.save(dest_path+filename+'_'+"rot20"+'_'+str(i))
                img_rot_30.save(dest_path+filename+'_'+"rot30"+'_'+str(i))
                i += 1


def image_resize(src_path,classe,size=128,dest_path=None,save=False):
    directory = os.listdir(src_path+classe+'/')
    if type(size) is not int:
        size = int(size)
    print(len(directory))
    images = np.empty([len(directory),size,size,3],dtype='uint8')
    for idx,image_name in list(enumerate(directory)):
        img = Image.open(src_path+classe+'/'+image_name)
        img = img.resize((size,size)).convert("RGB")
        if save or dest_path is not None:
            img.save(dest_path+"small_"+classe+"/"+classe+str(idx)+'.jpg')
        images[idx] = np.array(img)
    return images

def build_dataset(images,labels,test_ratio=0.1):
    np.random.seed(7)
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    images_train = images[int(test_ratio*len(images)):]
    labels_train = labels[int(test_ratio*len(images)):]
    images_dev = images[:int(test_ratio*len(images))]
    labels_dev = labels[:int(test_ratio*len(images))]
    return images_train,images_dev,labels_train,labels_dev


def save_h5_dataset(filename,path,images,labels):
    f = h5.File(path+filename+"_set"+".hdf5",'w')
    f.create_dataset("images_"+filename,data=images)
    f.create_dataset("labels_"+filename,data=labels)
    f.close()