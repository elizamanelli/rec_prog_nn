import numpy as np
import tensorflow as tf
from Prog_net.rnnprogressivenet import RNNProgressiveNet
from Data_Preparation.Data_Iterator import sequence_iterator
from tensorflow.contrib import rnn

# Make some fake observations.
inputSize = 23
fakeInputs1 = np.float64(np.random.rand(2000, 200, inputSize))
fakeTargets1 = np.reshape(np.repeat(np.random.randint(0, 9, 2000), 200, axis = 0), (2000, 200))
fakeLengths1 = np.random.randint(100, 200, 2000)
fakeTargets2 = np.reshape(np.repeat(np.random.randint(0, 7, 2000), 200, axis = 0), (2000, 200))
fakeLengths2 = np.random.randint(100, 200, 2000)
topology1 = [3, 2, 9]
topology2 = [2, 1, 7]
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
celltype = rnn.LSTMCell

it = sequence_iterator()
x = it.iter_data_epoch(fakeInputs1, fakeTargets1, 50, fakeLengths1)
y = it.iter_data_epoch(fakeInputs1, fakeTargets2, 50, fakeLengths2)

# create your progressive neural network with one extra column
progNet = RNNProgressiveNet(inputSize=inputSize, numLayers=3)
initCol = progNet.addColumn(topology1, activations, celltype,  [], logdir='/data/elli/random/firstcol',
                            batchnum=50, regression=False)
extCol = progNet.addColumn(topology2, activations, celltype, [initCol], logdir='data/elli/random/secondcol',
                           batchnum=50, regression=False)

initCol.create_optimizer()

for epoch in range(50):
    initCol.train(next(x), learning_rate=0.004, dropout_keep_prob=0.8)

    trainAccuracy = initCol.evaluate(next(x))

    msg = "InitialColumn Epoch = %d, train accuracy = %.2f%%"
    values = (epoch + 1, 100. * trainAccuracy[2])
    print(msg % values)

extCol.create_optimizer()

for epoch in range(50):
    extCol.train(next(y), learning_rate=0.004, dropout_keep_prob=0.8)

    trainAccuracy = extCol.evaluate(next(y))

    msg = "ExtendedColumn Epoch = %d, train accuracy = %.2f%%"
    values = (epoch + 1, 100. * trainAccuracy[2])
    print(msg % values)

progNet.writeToFile('/data/elli/random/savednet', epoch)