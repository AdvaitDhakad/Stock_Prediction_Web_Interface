# # from django.test import TestCase
# #
# # # Create your tests here.
# #
# # dict2 = {"Linear Regression": ["graph_3"], "Decision Tree": ["graph4"]}
# #
# # for ai, j in dict2.items():
# #     print(i,j[])
# # dict2 = {"Linear Regression": ["graph_3"], "Decision Tree": ["graph4"]}
# #
# # for k, (i, j) in enumerate(dict2.items()):
# #     print(j[k])
# import math
#
# import matplotlib.pyplot as plt
# import numpy
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import tensorflow as tf
# from sklearn.metrics import mean_squared_error
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.models import Sequential
#
#
# def fetching_price(stock="ADANIENT.NS"):
#     data = yf.download(tickers=stock, period='60d', interval='15m')
#     # data = yf.download(tickers=stock, period='720d', interval='1h')
#     # data = yf.download(tickers=stock, period='max', interval='1d')
#     df = pd.DataFrame(data)
#     # print(df.shape[0])
#     # df = df.Close
#         # .rolling(window='2D')
#     # prediction.predicting_1(df.shape[0])
#     # df = df.dropna()
#     # print(df.shape[0])
#     # print(df)
#     # print(data.head(10))
#     # print(data.tail(10))
#
#     return df
#
#
# def lstm(stock_name):
#     df = fetching_price(stock_name)
#     df1 = df.reset_index()['Close']
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
#     training_size = int(len(df1) * 0.65)
#     test_size = len(df1) - training_size
#     train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
#
#     def create_dataset(dataset, time_step=1):
#         dataX, dataY = [], []
#         for i in range(len(dataset) - time_step - 1):
#             a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
#             dataX.append(a)
#             dataY.append(dataset[i + time_step, 0])
#         return np.array(dataX), np.array(dataY)
#
#     time_step = 100
#     X_train, y_train = create_dataset(train_data, time_step)
#     X_test, ytest = create_dataset(test_data, time_step)
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
#     model.add(LSTM(50, return_sequences=True))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.summary()
#     model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=1, batch_size=64, verbose=1)
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)
#     train_predict = scaler.inverse_transform(train_predict)
#     test_predict = scaler.inverse_transform(test_predict)
#     math.sqrt(mean_squared_error(y_train, train_predict))
#     math.sqrt(mean_squared_error(ytest, test_predict))
#     look_back = 100
#     trainPredictPlot = numpy.empty_like(df1)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
#     # shift test predictions for plotting
#     testPredictPlot = numpy.empty_like(df1)
#     testPredictPlot[:, :] = numpy.nan
#     testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
#     # plot baseline and predictions
#     plt.title("Prediction using LSTM")
#     plt.plot(scaler.inverse_transform(df1))
#     plt.plot(trainPredictPlot)
#     plt.plot(testPredictPlot)
#     plt.legend(["Training", 'Validation', 'Prediction'])
#     plt.show()
#     plt.close()
#
#
#
# lstm("ADANIENT.NS")
# # fetching_price()

# import yfinance as yf
#
# stock_chosen = yf.Ticker("ADANIENT.NS")
# print(stock_chosen.info)

