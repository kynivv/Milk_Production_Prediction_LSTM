import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout


df = pd.read_csv('monthly-milk-production-pounds.csv', index_col='Month')
index = pd.to_datetime(df.index)
#df.plot()


scaler = MinMaxScaler()
array = []
train_data = []
train_labels = []

for i in range(len(df)):
    array.append(df.iloc[i]['Monthly milk production (pounds per cow)'])

array = np.array(array).reshape(-1,1)
array = scaler.fit_transform(array)

k = 0
for i in range(len(array)):
    try:
        train_data.append(array[12*k:12*(k+1)])
        train_labels.append(array[12*(k+1)])
        k+=1
    except:
        break

train_data = np.squeeze(train_data)
train_labels = np.array(train_labels)

#print(train_data.shape)


train_data = train_data[:len(train_labels)]
train_data = np.expand_dims(train_data,1)
#print(train_data.shape)


model = Sequential()

model.add(LSTM(250,input_shape=(1,12)))
model.add(Dropout(0.5))

model.add(Dense(250,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(12,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()


E = 1000

H = model.fit(train_data, train_labels, epochs=E)


epochs = range(0,E)
loss = H.history['loss']
#plt.plot(epochs,loss)


preds = scaler.inverse_transform(model.predict(train_data))
#plt.plot(range(0,13), preds, label='predictions')
#plt.plot(range(0,13), scaler.inverse_transform(train_labels),label='real values')
