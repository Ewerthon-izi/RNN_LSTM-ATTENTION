import pandas as pd
import numpy
import keras.backend as K
from google.colab import drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pandas import read_csv
from pandas import datetime
from pandas import Series
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Layer
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Input
from matplotlib import pyplot
from math import sqrt

#Constantes no codigo
LENGTH_PREVISION = -44
NUM_EPOCHS = 10
NUM_NEURONS = 32


# Classe de attention adaptada para LSTM
class attention(Layer):
    def __init__(self,return_sequences = True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='normal', trainable = True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable = True)        
        super(attention, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

#cria o modelo com attention
def create_fit_model_with_attention(train, batch_size, number_epochs, number_of_neurons):
  X, y = train[:,0:-1], train[:,-1]
  X = X.reshape(X.shape[0], 1, X.shape[1])
  model = Sequential()
  model.add(LSTM(number_of_neurons,return_sequences=True,
                 batch_input_shape=(batch_size, X.shape[1],
                                                X.shape[2]), 
                 stateful=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS))
  model.add(attention(return_sequences=True))
  model.add(Dense(NUM_NEURONS, activation='softmax'))
  model.compile(loss='mean_squared_error', optimizer='adam')

  for i in range(number_epochs):
    model.fit(X, y, epochs=25, batch_size=batch_size, verbose=0, shuffle=False)

  return model 

#cria o modelo sem attention
def create_fit_model(train, batch_size, number_epochs, number_of_neurons):
  X, y = train[:,0:-1], train[:,-1]
  X = X.reshape(X.shape[0], 1, X.shape[1])
  model = Sequential()
  model.add(LSTM(number_of_neurons,return_sequences=True,
                 batch_input_shape=(batch_size, X.shape[1],
                                                X.shape[2]), 
                 stateful=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS*2, return_sequences=True))
  model.add(LSTM(NUM_NEURONS))
  model.add(Dense(NUM_NEURONS, activation='softmax'))
  model.compile(loss='mean_squared_error', optimizer='adam')

  for i in range(number_epochs):
    model.fit(X, y, epochs=25, batch_size=batch_size, verbose=0, shuffle=False)

  return model

def forecast(model, batch_size, data_test):
  data_test = data_test.reshape(1, 1, len(data_test))
  yhat = model.predict(data_test, batch_size=batch_size)
  return yhat[0,0]


#cria dataframe
def timeseries_to_supervised(data, lag=1):
  df = DataFrame(data)
  columns = [df.shift(i) for i in range(1, lag+1)]
  columns.append(df)
  df = concat(columns, axis=1)
  df.fillna(0, inplace=True)
  return df

#-1 e 1
def scale(train, test):
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaler = scaler.fit(train)
  train_scaled = scaler.transform(train)
  test_scaled = scaler.transform(test)
  return scaler, train_scaled, test_scaled

def inverse_scale(scaler, test, yhat):
  new_row = [x for x in test] + [yhat]
  array = numpy.array(new_row)
  array = array.reshape(1, len(array))
  y_inverted = scaler.inverse_transform(array)
  return y_inverted[0, -1]

#tornar a serie estacionaria
def difference(dataset, interval=1):
  diff = list()
  for i in range(interval, len(dataset)):
    temp = dataset[i] - dataset[i-interval]
    diff.append(temp)
  return Series(diff)

def inverse_difference(original_dataset, yhat, pos=0):
  return yhat + original_dataset[pos]

def inverse_difference_series(original_dataset, difference_dataset, interval=1):
  inverted = list()
  for i in range(len(difference_dataset)):
    temp = inverse_difference(original_dataset, difference_dataset[i], 
                              i+interval-1)
    inverted.append(temp)
  return Series(inverted)


#ajustar os dados de mes e semana
def parser(x):
  return datetime.strptime(x,'%Y-%m-%W')

#faz a leitura do csv
drive.mount('/content/drive')
series = read_csv('/content/drive/My Drive/TCC/csv_tcc3.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#"seta" os valores
original_values = series.values
differenced = difference(series, 1)
supervised = timeseries_to_supervised(differenced)
supervised_values = supervised.values

#print(differenced.head())

#"seta" a base de dados
train, test = supervised_values[:LENGTH_PREVISION], supervised_values[LENGTH_PREVISION:]


scaler, train_scaled, test_scaled = scale(train, test)
model = create_fit_model_with_attention(train_scaled, 1, NUM_EPOCHS, NUM_NEURONS)

#preve as primeiras semanas
train_reshaped = train_scaled[:,0].reshape(len(train_scaled), 1, 1)
model.predict(train_reshaped, batch_size=1)

#preve as ultimas semanas
predictions = list()
for i in range(len(test_scaled)):
  X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
  yhat = forecast(model, 1, X)

  yhat = inverse_scale(scaler, X, yhat)

  yhat = inverse_difference(original_values, yhat, len(train_scaled)+i)

  predictions.append(yhat)
  expected = original_values[len(train) + i +1]

  print('Semana = %d, Previsto = %f, Esperado = %f' % (i+1, yhat, expected))

#erro médio
rmse = sqrt(mean_squared_error(original_values[LENGTH_PREVISION:], predictions))
print('RMSE: %.3f' % rmse)

#plota o grafico
pyplot.plot(original_values[LENGTH_PREVISION:])
pyplot.plot(predictions)
pyplot.show()

