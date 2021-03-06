# performing sequence prediction

# the following code uses a batch size of 9 for training and 1 for inference
# which brings the following error
# ---> ValueError: Input 0 is incompatible with layer sequential: expected shape=(9, None, 1), found shape=(1, 1, 1)

from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1)
# configure network
n_batch = len(X)
n_epoch = 1000
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
for i in range(n_epoch):
	model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = model.predict(testX, batch_size=1)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))

# ---> ValueError: Input 0 is incompatible with layer sequential: expected shape=(9, None, 1), found shape=(1, 1, 1)
