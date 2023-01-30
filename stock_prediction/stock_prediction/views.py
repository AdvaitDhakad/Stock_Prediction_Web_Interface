import os
import yahoo_fin.stock_info
# from django import forms
from django.http.response import HttpResponse
from django.shortcuts import render
import time
import queue
from threading import Thread
from yahoo_fin.stock_info import *
from . import analysis, prediction
import yfinance as yf


def home(request):
    return render(request, 'index.html', {})


def price_tracker(request):
    data = {}
    available_stocks = tickers_nifty50()
    # print(len(available_stocks))
    stockpicker = available_stocks[0:13]  # use 13
    n_threads = len(stockpicker)
    thread_list = []
    que = queue.Queue()
    start = time.time()
    count = 0
    for i in range(n_threads):
        thread = Thread(target=lambda q, arg1: q.put({stockpicker[i]: get_quote_table(arg1)}),
                        args=(que, stockpicker[i]))
        thread_list.append(thread)
        thread_list[i].start()
        print(count)
        count += 1

    for thread in thread_list:
        thread.join()
    while not que.empty():
        result = que.get()
        data.update(result)
    end = time.time()
    time_taken = end - start
    # print(time_taken)
    header_data = {}
    header_data_list = ['^NSEI', '^CNXIT', '^NSEBANK', '^BSESN']

    n_threads = len(header_data_list)
    thread_list = []
    que = queue.Queue()
    start = time.time()
    count = 0
    for i in range(n_threads):
        thread = Thread(target=lambda q, arg1: q.put({header_data_list[i]: get_quote_table(arg1)}),
                        args=(que, header_data_list[i]))
        thread_list.append(thread)
        thread_list[i].start()
        print(count)
        count += 1

    for thread in thread_list:
        thread.join()
    while not que.empty():
        result = que.get()
        header_data.update(result)
    end = time.time()
    time_taken = end - start
    # print(time_taken)
    stockchosen = request.POST.get('stock')
    if not stockchosen or stockchosen is None:
        stockchosen = 'HDFCLIFE.NS'
    print("stockchosen", stockchosen)
    if stockchosen not in available_stocks:
        return HttpResponse("Error NOT FOUND THE STOCK")

    def fetching_price(stock="ADANIENT.NS"):
        data = yf.download(tickers=stock, period='30d', interval='2m')
        # data = yf.download(tickers=stock, period='60d', interval='15m')
        # data = yf.download(tickers=stock, period='720d', interval='1h')
        # data = yf.download(tickers=stock, period='max', interval='1d')
        df = pd.DataFrame(data)
        return df

    stock_price = fetching_price(stockchosen)
    chart_1 = analysis.technical_analysis_1(stock_price, stockchosen)
    chart_2 = analysis.technical_analysis_2(stock_price, stockchosen)
    graph_3 = prediction.linear_regrex(stock_price)
    graph_4 = prediction.decision_tree(stock_price)
    graph_5 = prediction.lstm(stock_price)

    stock_chosen = yf.Ticker(stockchosen).info
    # print(stock_chosen)
    # time.sleep(5)
    # print(data)
    # print(header_data)
    dict1 = {'^NSEI': ['NIFTY', 'NSE.jpg'], '^CNXIT': ["NSE IT", 'NSEIT.jpg'], '^NSEBANK': ["NSE BANK", "bank.jpg"],
             '^BSESN': ["SENSEX", "BSE.png"]}
    dict2 = {"Linear Regression": [graph_3, "Linear regression is a statistical method used to model the linear "
                                            "relationship between a dependent variable and one or more independent "
                                            "variables. It is commonly used to make predictions about the value of "
                                            "the dependent variable based on the values of the independent variables. "
                                            "In a linear regression model, the dependent variable is assumed to be "
                                            "linearly related to the independent variables, and the model estimates "
                                            "the strength of this relationship using a slope coefficient for each "
                                            "independent variable. The slope coefficient represents the change in the "
                                            "dependent variable for each unit change in the independent variable, "
                                            "holding all other variables constant. Linear regression is a widely used "
                                            "method for modeling and understanding relationships in data, "
                                            "and it is particularly useful when the relationship between the "
                                            "variables is well-understood and the data is easy to work with."],
             "Decision Tree": [graph_4, "A decision tree is a machine learning algorithm that is used to classify "
                                        "items based on the values of certain features. It works by building a "
                                        "tree-like model of decisions, with each internal node representing a "
                                        "feature and each leaf node representing a classification. The algorithm "
                                        "starts at the root node and progresses through the tree by evaluating "
                                        "the features at each node and making a decision based on the values of "
                                        "those features. Decision trees are commonly used in classification tasks "
                                        "and can be effective at making predictions when the data has a clear "
                                        "structure. The tree is constructed by evaluating the features in the "
                                        "data and determining which feature is the most informative for making a "
                                        "prediction. The algorithm then splits the data based on the value of "
                                        "that feature, and the process is repeated for each subsequent split "
                                        "until a leaf node is reached. Decision trees can be used for a variety "
                                        "of applications, including classifying objects based on their "
                                        "characteristics, predicting customer behavior, and identifying patterns "
                                        "in data."],
             "Long Short Term Memory(LSTM)": [graph_5, """LSTM stands for Long Short-Term Memory, and it is a type of 
             artificial neural network specifically designed to process sequential data. LSTMs are used for tasks 
             such as language translation, speech recognition, and time series forecasting, where the order of the 
             data is important. LSTMs are able to learn long-term dependencies in data by using a special type of 
             memory called a cell state, which is able to retain information over long periods of time. The cell 
             state is modified by three types of gates, which control the flow of information into and out of the 
             cell state. This allows LSTMs to selectively remember or forget information, depending on the needs of 
             the task. LSTMs are a powerful tool for working with sequential data, and they have achieved 
             state-of-the-art results on a variety of tasks. However, they can also be computationally intensive and 
             difficult to train, particularly for large datasets."""]}

    return render(request, 'price_tracker.html',
                  {'data': data, 'stock_chosen': stock_chosen, 'chart': chart_1, 'chart_2': chart_2, 'graph_3': graph_3,
                   'graph_4': graph_4, "graph_5": graph_5, "header_data": header_data, "dict1": dict1, "dict2": dict2})
    # return render(request, 'price_tracker.html', {'data': data, 'stockchosen': stockchosen, "header_data": header_data})


def price_predictor(request):
    return render(request, 'price_predictor.html', {})
