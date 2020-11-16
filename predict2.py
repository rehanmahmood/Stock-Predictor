import requests
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, date
plt.style.use('fivethirtyeight')
today = date.today()


data = pd.read_csv('GC=F.csv', date_parser = True)


data_training = data[data['Date']<'2020-02-01'].copy()
data_test = data[data['Date']>='2020-02-01'].copy()

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)
print(data_training.tail(27))


scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)


X_train = []
y_train = []

for i in range(10, data_training.shape[0]):
    X_train.append(data_training[i-10:i])
    y_train.append(data_training[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)

regressior = Sequential()

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 120, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.compile(optimizer='adam', loss = 'mean_squared_error')

regressior.fit(X_train, y_train, epochs=27, batch_size=32)

pastDays = data_training.tail(27)

df = pastDays.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
print(df.head())








'''

df = web.DataReader('GC=F', data_source='yahoo', start='2019-02-14', end=str(today))
print(df.tail)

plt.figure(figsize=(16,8))
plt.title('GOLD PRICE HISTORY')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close price USD($)', fontsize=14)
#plt.show()
'''
