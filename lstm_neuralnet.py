import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries
import get_old_candle_data

# file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
input_file="DIS2.csv"

forecastCandle = 9
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back + forecastCandle, 3])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file, header=None, index_col=None, delimiter=',', usecols=[0,1,2,3])

lastTestData = get_old_candle_data.getOldCandleData()
df = np.array(df)
print('df_lstm_nn', df)

retrainingDataset = lastTestData

print('df length',len(df))
print(df[0:10])
# take close price column[5]
all_y = df
dataset=all_y.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

dataset=dataset.reshape(-1, 4)
print(dataset)

look_back = 240
# split into train and test sets, 50% test data, 50% training data
train_size = len(dataset)
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:]

# reshape into X=t and Y=t+1, timestep 240
print(len(train))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(len(trainX))
print(trainX)
print(len(testX))
print(len(testY))

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 4, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(25, input_shape=(4, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

#print(trainX[len(trainX)-1])
#print(testX[len(testX) - 1])
# print(len(testY))
# print(len(testPredict))
#print(testX[len(testX)-1])
# print(testY[0])
# print(scaler.inverse_transform([[0.04293486]]))
# print(scaler.inverse_transform([[0.04352662]]))
#print(scaler.inverse_transform([[0.04405421]]))
#print(scaler.inverse_transform([[0.044367921]]))



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1-(forecastCandle*2), :] = testPredict

# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#print('testPrices:')
#testPrices=scaler.inverse_transform(dataset[-(test_size - look_back + 1):-(test_size - look_back + 1 - (forecastCandle+1))])
# print(testPrices)
# print('testPredictions:')
# print(testPredict)
# print(len(testPredict))
# print(len(testPrices))
# print(train_size,train_size+(forecastCandle+1))
#testPrices = testPrices[train_size-look_back:train_size-look_back]
#print(len(testPrices))

# export prediction and actual prices
#df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
file_name = "lstm_result_5min_x_is_10_retraining2_debugging_2.csv" 
# df.to_csv(file_name, sep=';', index=None)
#df.to_json("testJson.json", orient = 'records')

# plot the actual price, prediction in test data=red line, actual price=blue line
#plt.plot(testPredictPlot)
#plt.show()
step = 10

def retrainingNN():
    arrX = lastTestData[-259:]
    #print(len(arrX))
    #print('retrainng_arrX',arrX)
    arrX = list(map(lambda x: [x], arrX))
    arrX = np.array(arrX)
    arrX = arrX.reshape(-1, 1)
    x = scaler.transform(arrX)
    #print(x)
    x = x.reshape(-1, 4)
    #print('retraining_x', x[-30:])
    dataX, dataY = [], []
    for i in range(10):
        a = x[i:(i+look_back)]
        dataX.append(a)
        dataY.append(x[i + look_back + forecastCandle, 3])
    #print('retrainng_dataX', dataX[-10:])
    #print('retrainng_dataY', dataY)
    trainX, trainY = np.array(dataX), np.array(dataY)
    # print('retraining_trainX', trainX[-5:])
    # print('retraining_trainy',trainY)
    trainX = np.reshape(trainX, (trainX.shape[0], 4, trainX.shape[1]))
    model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)



def appendLatestClose(latestCloseValue):
    lastTestData.append(latestCloseValue)



def predict_value(latestCloseValue):
    #print(lastTestData[len(lastTestData)-1])
    #print(lastTestData)
    arrX = lastTestData[-240:]
    #print(len(arrX))
    #print('predict_arrX',arrX)
    arrX = list(map(lambda x: [x], arrX))
    arrX = np.array(arrX)
    arrX = arrX.reshape(-1,1)
    x = scaler.transform(arrX)
    x = x.reshape(-1,4)
    # print('predict_x',x)
    a = x[0:(240)]
    a = np.array([a])
    # print('predict_a', a)
    b = np.reshape(a, (a.shape[0], 4, a.shape[1]))
    y = model.predict(b)
    y = scaler.inverse_transform(y)
    #prediction_Arr.append(y)
    #print('latest close value',latestCloseValue)
    #print('predicted value: ', y)
    return y[0][0]

#print(len(lastTestData))
# print(scaler.inverse_transform([[0.11311181]]))
# print(scaler.inverse_transform([[0.11335734]]))
# print(scaler.inverse_transform([[0.11335021]]))
retrainingNN()
predict_value(lastTestData[len(lastTestData)-1])
# for i in range(105121+step, dataset_len - step, step):
#     train_size = i
#     dataset_len = len(dataset) 
#     print(len(dataset))
#     test_size = len(dataset) - train_size + look_back
#     train, test = dataset[train_size-look_back-(forecastCandle+1+step):train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:]

#     # reshape into X=t and Y=t+1, timestep 240
#     print(len(train))
#     print(len(test))
#     #print(train[len(train)-20:])
#     #print(test[look_back+forecastCandle])
#     trainX, trainY = create_dataset(train, look_back)
#     testX, testY = create_dataset(test, look_back)
#     print(len(trainX))
#     print(len(testX))
#     print(len(testY))

#     # reshape input to be [samples, time steps, features]
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#     # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
#     #model = Sequential()
#     #model.add(LSTM(25, input_shape=(1, look_back)))
#     #model.add(Dropout(0.1))
#     #model.add(Dense(1))
#     #model.compile(loss='mse', optimizer='adam')
#     model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)

#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)

#     # invert predictions
#     trainPredict = scaler.inverse_transform(trainPredict)
#     trainY = scaler.inverse_transform([trainY])
#     testPredict = scaler.inverse_transform(testPredict)
#     testY = scaler.inverse_transform([testY])

#     #print(trainX[len(trainX)-1])
#     #print(testX[len(testX) - 1])
#     print(len(testY))
#     print(len(testPredict))
#     #print(testX[len(testX)-1])
#     print(testY[0])
#     #print(scaler.inverse_transform([[0.04293486]]))
#     #print(scaler.inverse_transform([[0.04352662]]))
#     #print(scaler.inverse_transform([[0.04405421]]))
#     #print(scaler.inverse_transform([[0.044367921]]))



#     # calculate root mean squared error
#     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore))

#     # shift train predictions for plotting
#     trainPredictPlot = np.empty_like(dataset)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#     # shift test predictions for plotting
#     testPredictPlot = np.empty_like(dataset)
#     testPredictPlot[:, :] = np.nan
#     testPrices=scaler.inverse_transform(dataset[-(test_size - look_back + 1):-(test_size - look_back + 1 - (forecastCandle+1))])
#     print(testPrices)
#     print('testPredictions:')
#     print(train_size,train_size+(forecastCandle+1))
#     print(len(testPrices))

#     # export prediction and actual prices
#     df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(testPrices.reshape(-1)), decimals=2)})
#     #file_name = "lstm_result_5min_x_is_10_retraining2"+ str(train_size)+ ".csv" 
#     df.to_csv(file_name, sep=';', mode = 'a', index=None, header = None)

#     # plot the actual price, prediction in test data=red line, actual price=blue line
#     #plt.plot(testPredictPlot)
#     #plt.show()

