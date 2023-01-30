import base64
import os.path
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from io import BytesIO
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
# from . import prediction
matplotlib.use('Agg')


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode("utf-8")
    buffer.close()
    return graph


def technical_analysis_1(stock_price, stock_name):
    # MOVING AVG
    stock = stock_price
    stock_mo = stock['Close'].to_frame()
    stock_mo['SMA30'] = stock['Close'].rolling(20).mean()
    print("done 1")
    stock_mo.dropna(inplace=True)
    # plt.figure(figsize=(15, 6))
    plt.ylabel("price")
    plt.xlabel("date")
    plt.title("MOVING AVG")
    plt.plot(stock_mo.tail(8).index, stock_mo['Close'].tail(8), color='blue',)
    plt.plot(stock_mo.tail(8).index, stock_mo['SMA30'].tail(8), color='red')
    print("done 2")
    plt.legend(["close price", "SMA30"], loc="upper right")
    # plt.savefig("D:/Languages/Programming_backed/pythonProject/Frontend-Minor/stock_prediction/home/static/fig1.png")
    graph = get_graph()
    plt.close()
    return graph


def technical_analysis_2(stock_price, stock_name):
    stock = stock_price
    # RSI
    delta = stock['Adj Close'].diff(1)
    delta.dropna(inplace=True)
    delta_positive = delta.copy()
    delta_negative = delta.copy()
    delta_positive[delta_positive < 0] = 0
    delta_negative[delta_negative > 0] = 0
    days = 14
    avg_gain = delta_positive.rolling(window=days).mean()
    avg_loss = abs(delta_negative.rolling(window=days).mean())
    relative_strength = avg_gain / avg_loss
    RSI = 100 - (100.0 / (1.0 + relative_strength))
    combined = pd.DataFrame()
    combined['Adj Close'] = stock['Adj Close']
    combined['RSI'] = RSI
    combined.dropna(inplace=True)
    plt.xlabel("DATE")
    plt.ylabel("PRICE")
    plt.title("RSI")
    # plt.plot(combined.index, combined['Adj Close'], color='green')
    plt.plot(combined.tail(8).index, combined['RSI'].tail(8), color='red')
    plt.axhline(0, linestyle="--", alpha=0.5, color='red')
    plt.axhline(30, linestyle="--", alpha=0.5, color='green')
    plt.axhline(70, linestyle="--", alpha=0.5, color='green')
    plt.axhline(100, linestyle="--", alpha=0.5, color='red')
    plt.legend(["RSI"], loc="upper right")

    # plt.savefig("D:/Languages/Programming_backed/pythonProject/Frontend-Minor/stock_prediction/home/static/RSI.png")
    print("figure2 saved")
    graph = get_graph()
    print("figure1 saved new")
    plt.close()
    return graph

