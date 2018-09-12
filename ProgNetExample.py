import numpy as np
import tensorflow as tf
from Prog_net.progressivenet import ProgressiveNeuralNetwork
from Data_Preparation.Data_Iterator import data_iterator

# Make some fake observations.
inputSize = 128
fakeInputs1 = np.float64(np.random.rand(4000, inputSize))
fakeTargets1 = np.eye(9)[np.random.randint(0, 9, 4000)]
fakeTargets2 = np.eye(7)[np.random.randint(0, 7, 4000)]
topology1 = [100, 64, 25, 9]
topology2 = [68, 44, 19, 7]
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]

it = data_iterator()
x = it.iter_data_epoch(fakeInputs1, fakeTargets1, 50)
y = it.iter_data_epoch(fakeInputs1, fakeTargets2, 50)

# create your progressive neural network with one extra column
progNet = ProgressiveNeuralNetwork(inputSize=inputSize, numLayers=4)
initCol = progNet.addColumn(topology1, activations, [], logdir='/data/elli/random/firstcol', regression=False)
extCol = progNet.addColumn(topology2, activations, [initCol], logdir='data/elli/random/secondcol', regression=False)

initCol.create_optimizer()

for epoch in range(50):
    initCol.train(next(x), learning_rate=0.004, dropout_keep_prob=0.8)

    trainAccuracy = initCol.evaluate(next(x), test=False)

    msg = "InitialColumn Epoch = %d, train accuracy = %.2f%%"
    values = (epoch + 1, 100. * trainAccuracy)
    print(msg % values)

extCol.create_optimizer()

for epoch in range(50):
    extCol.train(next(y), learning_rate=0.004, dropout_keep_prob=0.8)

    trainAccuracy = extCol.evaluate(next(y), test=False)

    msg = "ExtendedColumn Epoch = %d, train accuracy = %.2f%%"
    values = (epoch + 1, 100. * trainAccuracy)
    print(msg % values)

progNet.writeToFile('/data/elli/random/savednet', epoch)