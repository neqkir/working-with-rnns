We learn from https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/ 

By default Tensorflow Keras reset LSTM states to zero for each batch. We might want to implement another state strategy, see e.g., there https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html.

E.g., we might want to initialize with noisy states or to learn initial states together with the other parameters. In short, we want to set Tensorflow/Keras parameters `stateful` for the LSTM layer (or GRU) to `True`.

With `Stateful=True`, inference is producing an error when using the same model with a different batch size. This is the case when we train a model on large batch sizes, say 64 or 128, and do inference with a batch size of 1, for, say, predicting the next character, in text generation.

We present a simple example inspired from https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/, predicting next value for a sequence of numbers.

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... ->

We rewrote it for Tensorflow 2.0. In particular, the way to save and load weights has changed. We have to save weights to a .h5 file and load them for populating the inference model.

The work around consists in 

(1) creating a new model for inference

(2) populating this model with the weights from training

-> and we can infer with a batch size of 1 while our training operated with a batch size of 9.
