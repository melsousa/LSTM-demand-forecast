from numpy.lib.npyio import DataSource
# Importando
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.read_excel('Dados.xlsx', sheet_name='Query result')

demand = df['Vendas'].values

# Normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
demand = scaler.fit_transform(demand.reshape(-1,1))

# Divisão em treinamento e teste
train_size = int(len(demand) * 0.8)
test_size = len(demand) - train_size
train, test = demand[0:train_size,:], demand[train_size:len(demand),:]

# Conversão dos dados para a forma (samples, time steps, features)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Definindo que o look_back, olhará 5 dias atrás 
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape dos dados para a forma (samples, time steps, features)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Criação do modelo
model = Sequential()
model.add(LSTM(230, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Treinamento do modelo
model.fit(x=trainX, 
          y=trainY, 
          shuffle=True,
          validation_data=(testX, testY), 
          epochs=250, 
          batch_size=1, 
          verbose=2)

#  Previsão para os proxímos 5 dias

for i in range(5):
  # Dia inicial da visualização
  dia_inicial = i
  
  # Primeiros 5 dias da série.
  last_5 = demand[dia_inicial:dia_inicial+5]
  last_5 = np.reshape(last_5, (1, 5, 1))
  
  # Do 6º ao 10º dia
  next_5 = demand[dia_inicial+5:dia_inicial+5+1]
  next_5 = scaler.inverse_transform(next_5)
  
  # Previsão do que seria do 6º ao 10º dia
  next_5_pred = model.predict(last_5)
  next_5_pred = scaler.inverse_transform(next_5_pred)
  
  # Transformação inversão para os últimos 5 dias.
  last_5 = demand[dia_inicial:dia_inicial+5]
  last_5 = scaler.inverse_transform(last_5)
  
  # Plota
  plt.plot(range(1,len(last_5)+1), last_5, label='Passado')
  
  if next_5.shape[0] > 0:
      plt.plot(range(len(last_5), len(last_5)+len(next_5)+1),np.concatenate((last_5[-1].reshape(-1,1),next_5)),label='Futuro Verdadeiro')
  
  if next_5_pred.shape[0] > 0:
      plt.plot(range(len(last_5), len(last_5)+len(next_5_pred)+1),np.concatenate((last_5[-1].reshape(-1,1),next_5_pred)),label='Futuro Predito')
  
  
  plt.xticks(range(1,2))
  plt.xlabel('Dia')
  plt.ylabel('Demanda')
  plt.legend()
  plt.title('Gráfico de Demanda x Dia ' + str(i+1))
  plt.show()
