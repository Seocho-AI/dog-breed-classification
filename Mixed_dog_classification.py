from urllib.parse import urlencode, unquote, parse_qsl
from flask import Flask, json
from datetime import datetime
import pymysql
import requests
import math
# import pprint

app = Flask(__name__)
db = pymysql.connect(host='abandoned-dogs.cdlurfzj5gl4.ap-northeast-2.rds.amazonaws.com', port=3306, user='kaist',
                     passwd='0916', db='abandoned_dog', charset="utf8")
cursor = db.cursor()


# 오늘 날짜
today = datetime.today().strftime('%Y%m%d')
todayYear = today[:4]

# url 입력
url = "http://apis.data.go.kr/1543061/abandonmentPublicSrvc/abandonmentPublic?"

#queryString 입력
queryString = urlencode(
  {
    "serviceKey": unquote("ieHW%2FCmVoKfe3X9EnL2OT8JoMTqCSRxMT9%2FE5Fr4spuLN4s4Hms5ZiIAZm%2BgvmlMkm06BDRPZHKyrNW4Qo%2F%2B2w%3D%3D"),
    "bgnde": "20211001", # 20190101
    "endde": "20220101", # today
    "upkind": "417000", # 417000 = 개
    "kind": "000115", # 000114 = 믹스견 000115 = 기타
    "state": "", # protect = 보호 / notice = 공고 / null = 전체
    "pageNo": "1",
    "numOfRows": "10",
    "_type": "json"
  }
)

# API 파라미터 딕셔너리 형태 저장
queryDict = dict(parse_qsl(queryString))

# 최종 요청 url 생성
queryURL = url + queryString

# API 호출
response = requests.get(queryURL)

# 딕셔너리 형태로 변환
r_dict = json.loads(response.text)

# 오픈 API 호출 결과 데이터의 개수 확인 및 저장
numOfRows = r_dict["response"]["body"]["numOfRows"]

# 전체 데이터의 개수 확인 및 저장
totCnt = r_dict["response"]["body"]["totalCount"]

# 총 오픈 API 호출 횟수 계산 및 저장
loopCount = math.ceil(totCnt/numOfRows)

# 완전한 response element 리스트
# completeResponseEl = ["desertionNo", "filename", "happenDt", "happenPlace", "kindCd", "colorCd", "age", "weight", "noticeNo", "noticeSdt", "noticeEdt", "popfile", "processState", "sexCd", "neuterYn", "specialMark", "careNm", "careTel", "careAddr", "orgNm", "officetel"]

print(totCnt)
# open API 호출을 총 오픈 API 호출 횟수만큼 반복 실행



#----------------------------------------------------model------------------------------------------------------#

import tensorflow as tf

import os
import gc

from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd

from keras import regularizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception, preprocess_input
import torch

import urllib
from urllib import request
from io import BytesIO
from PIL import Image

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = tf.keras.Input(input_size)
    preprocessor = tf.keras.layers.Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    feature_extractor = tf.keras.Model(inputs = input_layer, outputs = avg)
    
    #Extract feature.
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps

def extact_features(data):
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)

    print('Final feature maps shape', final_features.shape)

    #deleting to free up ram memory
    del inception_features 
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()


    return final_features



img_size = (331,331,3)

inception_preprocessor = preprocess_input
xception_preprocessor = preprocess_input
inc_resnet_preprocessor = preprocess_input
nasnet_preprocessor = preprocess_input

loaded_model_tf = tf.keras.models.load_model('./classification_model.h5')
labels = pd.read_csv('./labels.csv')
classes = sorted(list(set(labels['breed'])))



# ======================================================================================= #
# 이미 분류되어 db에 들어가있는 개체 제외
import pandas as pd

sql = "select desertionNo from dog_list where mixPredict is null"
df = pd.read_sql(sql,db)
# print(df.shape)
# print(df.head())
# ======================================================================================= #

#---------------------------------------------model-loop--------------------------------------------------------#

for i in range(loopCount):
  queryDict["pageNo"] = i + 1                         # 페이지 수만큼 pageNo 증가
  queryString = urlencode(queryDict)                  # 다시 queryString 업데이트
  queryURL = url + queryString                        # 최종 요청 url 생성
  response = requests.get(queryURL)                   # API 호출
  r_dict = json.loads(response.text)                  # 딕셔너리 형태로 변환
  dogs = r_dict["response"]["body"]["items"]["item"]  # "dogs" is a list and contains 1000 individual dog info as dict
  mix_predict = []
  for dog in dogs:                                    # iterate through each "dog" in "dogs"
    img_url = dog['popfile']
    desertionNo = dog['desertionNo']

    if (df['desertionNo'] == desertionNo).any() :
        print(desertionNo+' start')
    else: 
        print(desertionNo + ' already classified')
        continue
    
    try:
      res = request.urlopen(img_url).read()
    except:
      print(f'{desertionNo} failed')
      continue
    img_g = Image.open(BytesIO(res)).resize(img_size[:2]).convert('RGB')
    
    img_g = np.expand_dims(img_g, axis=0) # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format

    # img_g
    # #Predict test labels given test data features.
    test_features = extact_features(img_g)
    predg = loaded_model_tf.predict(test_features)
    # print(f"Predicted label: {classes[np.argmax(predg[0])]}")
    # print(f"Probability of prediction): {round(np.max(predg[0]),2) * 100} %")

    tensor=torch.tensor(predg)

    val, idx = torch.sort(tensor,  descending=True)
    # 값 top2 값 매핑
    # print(val)

    # 값 true인 index들 - prob 높은것부터
    index = ((tensor>0.1).squeeze() == True).nonzero(as_tuple=True)[0]
    index = tensor.squeeze().argsort(descending=True)[:len(index)]

    mix_prob = {} # {'desertionNo':desertionNo, 'mixPredict':[1st_breed, 1st_prob, 2nd_breed, 2nd_prob, ...]}
    mix_prob['desertionNo'] = desertionNo
    mix_prob['mixPredict'] = []
    for j in index:
      breed = classes[j]
      prob = round((tensor[0][j].item())*100,1)
    #   print(breed, prob)
      mix_prob['mixPredict'].append(breed)
      mix_prob['mixPredict'].append(prob)
    print(mix_prob)
    # mixPredict값을 string형태로 넣어 db update
    mix_prob['mixPredict'] = str(mix_prob['mixPredict'])
    mix_predict.append(mix_prob)
  
  # pprint.pprint(mix_predict)
  
  sql = "UPDATE dog_list SET kindCd = '믹스견', mixPredict = %(mixPredict)s WHERE desertionNo = %(desertionNo)s;"
  cursor.executemany(sql, mix_predict)
  db.commit()
  print(i + 1, "번째", cursor.rowcount, "record inserted.")