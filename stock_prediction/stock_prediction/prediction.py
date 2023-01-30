from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import mpld3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error


def decision_tree(stock_name):
    stock_name = stock_name.tail(30)
    X = stock_name.drop(['Close'], axis=1)
    y = stock_name['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Plot the actual prices vs predicted prices
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Stock Price Prediction')
    graph_4 = mpld3.fig_to_html(fig)
    return graph_4


def linear_regrex(stock_name):
    stock_name = stock_name.tail(30)
    X = stock_name.drop(['Close'], axis=1)
    y = stock_name['Close']
    X_train = X[:int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    y_train = y[:int(len(y) * 0.8)]
    y_test = y[int(len(y) * 0.8):]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Plot the actual prices vs predicted prices
    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Stock Price Prediction')
    graph_3 = mpld3.fig_to_html(fig)
    # graph = get_graph()
    return graph_3


def lstm(stock_name):
    df = stock_name
    df1 = df.reset_index()['Close']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=1, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    math.sqrt(mean_squared_error(y_train, train_predict))
    math.sqrt(mean_squared_error(ytest, test_predict))
    look_back = 100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # plot baseline and predictions
    fig = plt.figure(figsize=(12, 6))
    plt.title("Prediction using LSTM")
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.legend(["Training", 'Validation', 'Prediction'])
    graph_5 = mpld3.fig_to_html(fig)
    plt.close()
    return graph_5



# # available_stocks = tickers_nifty50()
# # print(len(available_stocks))
# # df = pd.read_csv("D:/DOWNLOADS/data.csv")
# # # print([df['Symbol']])
# # list = df['Symbol'].to_list()
# # print(list)
# # def predicting_1(stock_name):
# #     print("printing from predicting_1")
# #     print(stock_name)
# #     print(stock_name)
# # # ['NIFTY 50.NS', 'ADANIENT.  NS', 'TCS.NS', 'RELIANCE.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'ULTRACEMCO.NS', 'ICICIBANK.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'SBIN.NS', 'ONGC.NS', 'DIVISLAB.NS', 'HCLTECH.NS', 'INFY.NS', 'TATACONSUM.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'ITC.NS', 'BAJAJFINSV.NS', 'HDFC.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'BAJAJ-AUTO.NS', 'JSWSTEEL.NS', 'TITAN.NS', 'KOTAKBANK.NS', 'COALINDIA.NS', 'GRASIM.NS', 'BPCL.NS', 'HINDALCO.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'DRREDDY.NS', 'BAJFINANCE.NS', 'TATASTEEL.NS', 'TECHM.NS', 'POWERGRID.NS', 'HDFCLIFE.NS', 'ADANIPORTS.NS', 'NTPC.NS', 'BRITANNIA.NS', 'MARUTI.NS', 'LT.NS', 'M&M.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'UPL.NS', 'EICHERMOT.NS', 'SBILIFE']
# import yfinance as yf
#
# # Get information about a stock
# # stock = yf.Ticker("AAPL")
# stockchosen = "ADANIENT.NS"
# stock_chosen = yf.Ticker(stockchosen).info
# # print(stock_chosen)
#
# # Print the stock info
# print(stock_chosen)
# #
# # # Get information about a stock
# # info = yahoo_fin.stock_info.ticker_info(stockchosen)
# #
# # # Print the stock info
# # # print(info)
# header_data_list = ['^NSEI', '^CNXIT', '^NSEBANK', '^BSESN']
# dict1 = {'^NSEI': ['NSE', 'NSE.jpg'], '^CNXIT': ["NSE IT", 'NSEIT.jpg'], '^NSEBANK': ["NSE BANK", "bank.jpg"],
#          '^BSESN': ["SENSEX", "BSE.png"]}
# print(dict1.get(header_data_list[1])[0])