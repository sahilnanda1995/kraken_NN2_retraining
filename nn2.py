# example making new probability predictions for a classification problem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# generate 2d classification dataset

input_file="nn2examples.csv"

entry_test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[0])
prediction = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[1])
test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[2])
volume = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[3])

arr = []

entry_test_price = entry_test_price.values
prediction = prediction.values
test_price = test_price.values
volume = volume.values

scaler_volume = MinMaxScaler(feature_range=(0, 1))
volume = scaler_volume.fit_transform(volume)

scaler_predProfit = MinMaxScaler(feature_range=(0, 1))

arrX = []
arrY = []
actionArr = []
predProfitArr = []
volumeArr = []

for i in range(0,len(volume),10):
    volumeArr.append(volume[i][0])

for i in range(0,len(entry_test_price),10):
    predProfit = float(prediction[i][0] - entry_test_price[i][0])/entry_test_price[i][0] * 100
    # print('predicted profit', predProfit)
    if predProfit >= 0:
        action = 1
    else:
        action = 0
        predProfit = abs(predProfit)
    predProfitArr.append(predProfit)
    actionArr.append(action)
    actualProfit = float(test_price[i][0] - entry_test_price[i][0])/entry_test_price[i][0] * 100
    # print('actual profit', actualProfit)
    if action == 1:
        if actualProfit >= 0.2:
            arrY.append(1)
        else:
            arrY.append(0)
    elif action == 0:
        if actualProfit <= -0.2:
            arrY.append(1)
        else:
            arrY.append(0)


print('volume length', len(volume))
predProfitArr = np.array(predProfitArr)
predProfitArr = predProfitArr.reshape(-1,1)
predProfitArr = scaler_predProfit.fit_transform(predProfitArr)
# print(predProfitArr)
actionArr = np.array(actionArr)
actionArr = actionArr.reshape(-1,1)
print(len(volumeArr))
arrY = np.array(arrY)
arrY = arrY.reshape(-1,1)
print(arrY)
print(len(arrY))

for i in range(len(predProfitArr)):
    arrX.append(predProfitArr[i][0])
    arrX.append(actionArr[i][0])
    arrX.append(volumeArr[i])

print(len(arrX))
# print(arrX)

arrX = np.array(arrX)
arrX = arrX.reshape(-1,3)
# print(arrX)
# print(X)
# print(y)
# scalar = MinMaxScaler()
# scalar.fit(X)
# print(X)
# X = scalar.transform(X)

# X = [0.89,0.1,0.95,0.3,0.79,0.2,0.12,0.7,0.33,0.9,0.22,0.89]
# X = np.array(X)
# X = X.reshape(-1,2)
# y = [1,1,1,0,0,0]
# y = np.array(y)



# print(X)
# define and fit the final model


model = Sequential()
model.add(Dense(25, input_dim=3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(arrX, arrY, epochs=64, verbose=1)
# new instances where we do not know the answer

ynew = model.predict_proba(arrX[0:30])

for i in range(0,30):
    print("X=%s, Predicted=%s, arrY=%s" % (arrX[i], ynew[i], arrY[i]))


# model.save('nn2.h5')  # creates a HDF5 file 'my_model.h5'

def predict_probab_not_skipping(trainY, prediction, volumeX):
    # print('Inside predict_value')
    # print(trainY)
    # print(prediction)
    # print(volumeX)
    trainY = np.array(trainY)
    trainY = trainY.reshape(-1, 1)
    prediction = np.array(prediction)
    prediction = prediction.reshape(-1, 1)
    volumeX = np.array(volumeX)
    volumeX = volumeX.reshape(-1, 1)
    volumeX = scaler_volume.transform(volumeX)
    predProfitX = []
    actionX = []
    trainX = []
    for i in range(len(trainY)):
        predProfit = float(prediction[i][0] - trainY[i][0])/trainY[i][0] * 100
        # print('predicted profit', predProfit)
        if predProfit >= 0:
            action = 1
        else:
            action = 0
            predProfit = abs(predProfit)
        predProfitX.append(predProfit)
        actionX.append(action)
    # print(predProfitX)
    # print(actionX)
    actionX = np.array(actionX)
    actionX = actionX.reshape(-1, 1)
    predProfitX = np.array(predProfitX)
    predProfitX = predProfitX.reshape(-1, 1)
    predProfitX = scaler_predProfit.transform(predProfitX)
    # print('num', predProfitX)
    # print('num', actionX)
    for i in range(len(predProfitX)):
        trainX.append(predProfitX[i][0])
        trainX.append(actionX[i][0])
        trainX.append(volumeX[i][0])
    # print(trainX)
    trainX = np.array(trainX)
    trainX = trainX.reshape(-1,3)
    # print(trainX)
    predProb = model.predict_proba(trainX)
    # print('inside predval', predProb)
    return predProb[0][0]


    

# model = load_model('my_model.h5')
