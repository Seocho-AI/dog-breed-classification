#Importing required libraries

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc

from sklearn.model_selection import train_test_split


import tensorflow as tf
from tqdm.autonotebook import tqdm

import numpy as np #
import pandas as pd

from keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.utils import to_categorical
# from keras import regularizers
from tensorflow.keras.models import Model


from tensorflow.keras.preprocessing.image import load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def barw(ax):
    for p in ax.patches:
        val = p.get_width()  # height of the bar
        x = p.get_x() + p.get_width()  # x- position
        y = p.get_y() + p.get_height() / 2  # y-position
        ax.annotate(round(val, 2), (x, y))

def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    # Extract feature.
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


def images_to_array_test(test_path, img_size=(331, 331, 3)):
    test_filenames = [test_path + fname for fname in os.listdir(test_path)]

    data_size = len(test_filenames)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for ix, img_dir in enumerate(tqdm(test_filenames)):
        #         img_dir = os.path.join(directory, image_name + '.jpg')
        img = load_img(img_dir, target_size=img_size)
        #         img = np.expand_dims(img, axis=0)
        #         img = processed_image_resnet(img)
        #         img = img/255
        images[ix] = img
        #         images[ix] = img_to_array(img)
        del img
    print('Ouptut Data Size: ', images.shape)
    return images


def extact_features(data):
    inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
    xception_features = get_features(Xception, xception_preprocessor, img_size, data)
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, img_size, data)
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features], axis=-1)

    print('Final feature maps shape', final_features.shape)

    # deleting to free up ram memory
    del inception_features
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()

    return final_features
# from keras.preprocessing.image import img_to_array
# from keras.applications.imagenet_utils import decode_predictions

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

labels = pd.read_csv('labels.csv')

#Create list of alphabetically sorted labels.
classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))



#Map each label string to an integer label.
class_to_num = dict(zip(classes, range(n_classes)))

input_shape = (331,331,3)
def images_to_array(directory, label_dataframe, target_size=input_shape):
    image_labels = label_dataframe['breed']
    images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]],
                      dtype=np.uint8)  # as we have huge data and limited ram memory. uint8 takes less memory
    y = np.zeros([len(label_dataframe), 1], dtype=np.uint8)

    for ix, image_name in enumerate(tqdm(label_dataframe['id'].values)):
        img_dir = os.path.join(directory, image_name + '.jpg')
        img = load_img(img_dir, target_size=target_size)
        #         img = np.expand_dims(img, axis=0)
        #         img = processed_image_resnet(img)
        #         img = img/255
        images[ix] = img
        #         images[ix] = img_to_array(img)
        del img

        dog_breed = image_labels[ix]
        y[ix] = class_to_num[dog_breed]

    y = to_categorical(y)

    return images, y

import time
t = time.time()

X,y = images_to_array('train', labels[:])

print('runtime in seconds: {}'.format(time.time() - t))

# np.where(y[5]==1)[0][0]

# lets check some dogs and their breeds
n = 25

# setup the figure
plt.figure(figsize=(20, 20))

for i in range(n):
    #     print(i)
    ax = plt.subplot(5, 5, i + 1)
    plt.title(classes[np.where(y[i] == 1)[0][0]])
    plt.imshow(X[i].astype('int32'))  # .astype('int32') ---> as imshow() needs integer data to read the image

#Learning Rate Annealer
lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5,verbose = 1)

#Prepare call backs
EarlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Hyperparameters
batch_size= 128
epochs=50
learn_rate=.001
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
adam=Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None,  amsgrad=False)

img_size = (331,331,3)

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)

from keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)

# Extract features using InceptionResNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)

# Extract features using NASNetLarge
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge,
                               nasnet_preprocessor,
                               img_size, X)

del X #to free up some ram memory
gc.collect()

#Creating final featuremap by combining all extracted features

final_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features,], axis=-1) #axis=-1 to concatinate horizontally

print('Final feature maps shape', final_features.shape)

#Prepare Deep net

model = Sequential()
# model.add(Dense(1028,input_shape=(final_features.shape[1],)))
model.add(Dropout(0.7,input_shape=(final_features.shape[1],)))
model.add(Dense(n_classes,activation= 'softmax'))

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Training the model.
history = model.fit(final_features, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[lrr,EarlyStop])

#deleting to free up ram memory

del inception_features
del xception_features
del nasnet_features
del inc_resnet_features
del final_features
gc.collect()

test_data = images_to_array_test('test/', img_size)

test_features = extact_features(test_data)

#Free up some space.
del test_data
gc.collect()

#Predict test labels given test data features.

pred = model.predict(test_features)
import pickle
with open("clasification_model","wb") as f:
    pickle.dump(model,f)
# First prediction
print(pred[0])
print(f"Max value (probability of prediction): {np.max(pred[0])}") # the max probability value predicted by the model
print(f"Sum: {np.sum(pred[0])}") # because we used softmax activation in our model, this will be close to 1
print(f"Max index: {np.argmax(pred[0])}") # the index of where the max value in predictions[0] occurs
print(f"Predicted label: {classes[np.argmax(pred[0])]}")

# Create pandas DataFrame with empty columns
preds_df = pd.DataFrame(columns=["id"] + list(classes))
preds_df.head()

# Append test image ID's to predictions DataFrame
test_path = "test/"
preds_df["id"] = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
preds_df.head()

preds_df.loc[:,list(classes)]= pred

preds_df.to_csv('submission.csv',index=None)
preds_df.head()

#Custom input

#Image('dataset/[경기-수원-2022-00328].jpg')

#reading the image and converting it into an np array

# img_g = load_img('dataset/[경기-수원-2022-00328].jpg',target_size = img_size)
# img_g = np.expand_dims(img_g, axis=0) # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format
# # img_g
#
# # #Predict test labels given test data features.
# test_features = extact_features(img_g)
# predg = model.predict(test_features)
# print(f"Predicted label: {classes[np.argmax(predg[0])]}")
# print(f"Probability of prediction): {round(np.max(predg[0])) * 100} %")