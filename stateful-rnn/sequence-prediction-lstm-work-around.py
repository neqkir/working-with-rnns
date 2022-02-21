# > this time we can train and infer with different batch sizes
# > (1) we build a brand new model for inference
# > (2) we load training weights and generate in the inference model

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
n_batch = 3
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

#saving weights
model.save_weights("sequence_pred.h5")

# re-define the batch size
n_batch = 1
# re-define model
new_model = Sequential()
new_model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
new_model.add(Dense(1))
# copy weights
new_model.load_weights("sequence_pred.h5")
# compile model
new_model.compile(loss='mean_squared_error', optimizer='adam')
# online forecast
for i in range(len(X)):
	testX, testy = X[i], y[i]
	testX = testX.reshape(1, 1, 1)
	yhat = new_model.predict(testX, batch_size=n_batch)
	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
