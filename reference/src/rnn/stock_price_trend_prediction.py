# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset_train = pd.read_csv('data/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set_scaled = scaler.fit_transform(training_set)

# create data structure with 60 time steps. Less time step more over fitting.
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# reshape to match keras docs for RNN input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer="adam", loss="mean_squared_error")
regressor.fit(X_train, y_train, epochs=100, batch_size=32)



dataset_test = pd.read_csv('data/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
real_stock_prices = dataset_test.iloc[:, 1:2].values

dataset_full = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_full[len(dataset_full) - len(dataset_test) - 60:].values

inputs = np.reshape(inputs, (-1, 1))
inputs = scaler.transform(inputs)

regressor.save("model.h5")

