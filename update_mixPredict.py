from flask import Flask
from datetime import datetime, timedelta
import pymysql
from threading import Timer

import tensorflow as tf

import gc

import numpy as np
import pandas as pd

# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.nasnet import NASNetLarge, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception, preprocess_input
import torch

from urllib.request import urlopen
from io import BytesIO
from PIL import Image


# ------------------------------ Model Load ------------------------------ #

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


# --------------------------- Connect PyMySQL --------------------------- #

app = Flask(__name__)
db = pymysql.connect(host='abandoned-dogs.cdlurfzj5gl4.ap-northeast-2.rds.amazonaws.com', port=3306, user='kaist',
                     passwd='0916', db='abandoned_dog', charset="utf8")
cursor = db.cursor()


# ------------------- GETTING TODAY'S YEAR/MONTH/DATE ------------------- #

today = datetime.strftime(datetime.now() - timedelta(1), '%Y%m%d')  # 어제
two_weeks_before = datetime.strftime(datetime.now() - timedelta(15), '%Y%m%d')  # 어제 - 14


# ------------------- DB에서 update할 개체정보 불러오기 ---------------------- #
                   
sql = "select desertionNo, popfile from dog_list where (happenDt>=%s) and kindCd='믹스견' and processState='보호중' and mixPredict is null;"
cursor.execute(sql,two_weeks_before)
rows = cursor.fetchall()


# ------------------------ 믹스견 분류 ------------------------ #

def update_mixPredict():
    
    # iterate through each "dog" in rows    
    for dog in rows:
        # 이미지url과 desertionNo 저장
        img_url = dog[1]
        desertionNo = dog[0]
    
        # 이미지url 요청 실패한 경우 pass
        try:
            res = urlopen(img_url).read()
        except:
            print(f'{desertionNo} failed')
            continue
        
        # image resize
        try:
            img_g = Image.open(BytesIO(res)).resize(img_size[:2]).convert('RGB')
        except:
            continue
        # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format
        img_g = np.expand_dims(img_g, axis=0)

        # Predict test labels given test data features.
        test_features = extact_features(img_g)
        predg = loaded_model_tf.predict(test_features)
        # print(f"Predicted label: {classes[np.argmax(predg[0])]}")
        # print(f"Probability of prediction): {round(np.max(predg[0]),2) * 100} %")

        tensor=torch.tensor(predg)

        # val, idx = torch.sort(tensor,  descending=True)
        # print(val)

        # 값 true인 index들 - prob 높은것부터
        index = ((tensor>0.1).squeeze() == True).nonzero(as_tuple=True)[0]
        index = tensor.squeeze().argsort(descending=True)[:len(index)]

        # "mix_prob" : 개체별 예측값 담은 딕셔너리
        # {'desertionNo':desertionNo, 'mixPredict':[1st_breed, 1st_prob, 2nd_breed, 2nd_prob, ...]}
        mix_prob = {}
        mix_prob['desertionNo'] = desertionNo
        mix_prob['mixPredict'] = []
        for j in index:
            breed = classes[j]
            prob = round((tensor[0][j].item())*100,1)
            #   print(breed, prob)
            mix_prob['mixPredict'].append(breed)
            mix_prob['mixPredict'].append(prob)
        print(mix_prob)

        # mixPredict값을 string형태로 넣어 DB update
        mix_prob['mixPredict'] = str(mix_prob['mixPredict'])
        
        # DB에 넣기 위해 쿼리 실행
        sql = "UPDATE dog_list SET mixPredict = %(mixPredict)s WHERE desertionNo = %(desertionNo)s;"
        cursor.execute(sql, mix_prob)
        db.commit()
    print('committed!')

    # 24시간마다 반복
    Timer(86400, update_mixPredict).start()
        
update_mixPredict()



