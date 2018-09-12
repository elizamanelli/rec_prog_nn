import tensorflow as tf
import numpy as np
from os.path import dirname
import pickle
import os
from Prog_net.progressivenet import Column, ProgressiveNeuralNetwork
from tensorflow.contrib import rnn

#Helper functions

def weight_variable(shape, name, stddev=0.4, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float64)
    if 'U' in name:
        with tf.name_scope('UWeight'):
            return tf.Variable(initial, name=name)
    with tf.name_scope('Weight'):
        return tf.Variable(initial, name=name)

def bias_variable(shape, name, init_bias=0.1, initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float64)
    with tf.name_scope('Bias'):
        return tf.Variable(initial, name=name)


class Rnn_Column(Column):
    '''An extensible network column for use in transfer learning with a Progressive
    Neural Network using recurrent structures.'''

    def __init__(self, topology: list, activations: list, celltype,
                 session, previousColumns: list, colIndex: int, logdir: str, batchsize: int, dtype=tf.float64,
                 regression = False
    ) -> None:
        '''
        :param topology: list containing the number of units per layer.
        :param activations: list containing the activation functions of those layers.
        :param celltype: Specifying the type of RNN cell used (GRU, BasicLSTM, LSTM,...)
        :param session: tensorflow session to connect with this column
        :param previousColumns: list containing the previous columns of the prog net.
        :param colIndex: int index of the current column - used for naming
        :param logdir: str directory used for logging
        :param batchsize: int batchsize for first initialization - can be changed later on
        :param dtype: tf datatype to use

        Creating one column for a recurrent progressive neural net.
        '''

        def name(*args, **kwargs) -> str:
            return  self.name(*args, **kwargs)

        n_input = topology[0]
        self.regression = regression
        self.batchsize = batchsize
        self.colIndex = colIndex
        self.topology = topology
        self.session = session
        self.mergedperbatch = None
        self.mergedupdate = []
        self.mergedperepoch = []
        self.summarywritertrain = None
        self.summarywritertest = None

        width = len(previousColumns)

        # numLayers in network - first value doesn't count.

        numLayers = self.numLayers = len(topology) - 1
        prevCols = self.previousColumns = previousColumns

        #Columns must have the same height
        assert all([self.numLayers == col.numLayers for col in prevCols])
        with tf.name_scope(str(colIndex)):

            with tf.name_scope('batch_size'):
                self.batch_size = tf.placeholder_with_default(
                    np.int32(50), shape=(), name=name('batch_size')
                )

            with tf.name_scope('inputs'):
                #Shape = [Batchsize, Timesteps, Features per frame]
                self.inputs = tf.placeholder(
                    dtype, shape=[None, None, n_input], name=name('inputs')
                )
            with tf.name_scope('targets'):
                #Shape = [Batchsize, Timesteps]
                #For mtl learning make [Batchsize, Timesteps, 2]
                if self.regression:
                    self.targets = tf.placeholder(
                        dtype, shape=[None, None], name=name('targets')
                    )
                else:
                    self.targets = tf.placeholder(
                        tf.float32, shape=[None, None], name=name('targets')
                    )
            with tf.name_scope('learning_rate'):
                self.learning_rate = tf.placeholder_with_default(
                    np.float64(0.0), shape=(), name=name('learning_rate')
                )
            with tf.name_scope('length_list'):
                #Shape = [Batchsize]
                self.length_list = tf.placeholder(
                    dtype, shape=[None], name=name('length_list')
                )
            with tf.name_scope('dropout'):
                self.dropout_keep_prob = tf.placeholder_with_default(
                    np.float64(1.0), shape=(), name=name('dropout_keep_prob')
                )

            tf.add_to_collection(name('dropout_keep_prob'), self.dropout_keep_prob)
            tf.add_to_collection(name('learning_rate'), self.learning_rate)
            tf.add_to_collection(name('length_list'), self.length_list)

            self.W = [[]] * numLayers
            self.R = [[]] * numLayers
            self.b = [[]] * numLayers
            self.U = []
            for k in range(numLayers-1):
                self.U.append( [[]] * width )
            self.h = [self.inputs]
            params = []
            Wsum =[]

            for k in range(numLayers):
                with tf.name_scope(str(k)):
                    if k > 0:
                        with tf.variable_scope(str(colIndex)+str(k)+'rnn') as vs:
                            cell = celltype(topology[k])
                            cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.dropout_keep_prob)
                            value, _ = tf.nn.dynamic_rnn(
                                cell, self.h[-1], dtype=tf.float64,
                                sequence_length=self.length_list)
                            values_flat = tf.reshape(value, [-1, topology[k]])
                            self.R[k] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
                            params.extend(self.R[k])
                    else:
                        values_flat = tf.reshape(self.h[-1], [-1, topology[k]])
                    W_shape = topology[k:k+2]
                    Wk = self.W[k] = weight_variable(W_shape, name('W', k))
                    bk = self.b[k] = bias_variable([W_shape[1]], name('b', k))
                    if 'log' in logdir:
                        Wsum.append(tf.summary.histogram(name='Weights'+str(k), values=Wk))

                    with tf.name_scope('Preactivation'):
                        preactivation = tf.matmul(values_flat, Wk) + bk
                    params.append(Wk)
                    params.append(bk)
                    #variable_summaries(Wk)
                    #variable_summaries(bk)
                    if k > 0:
                        for kk in range(width):
                            U_shape = [prevCols[kk].topology[k], topology[k+1]]
                            Uk = self.U[k-1]
                            Uk[kk] = weight_variable(U_shape, name('U', k, kk))
                            if 'log' in logdir:
                                Wsum.append(tf.summary.histogram(name='UWeights'+str(k)+str(kk), values=Uk[kk]))
                            #variable_summaries(Uk[kk])
                            with tf.name_scope('Preactivation'):
                                preactivation += tf.matmul(
                                    tf.reshape(prevCols[kk].h[k], [-1, prevCols[kk].h[k].shape[-1]]), Uk[kk]
                                )
                        for kk in range(width):
                            params.append(self.U[k-1][kk])

                        with tf.name_scope('Activations'):
                            act = activations[k](preactivation)
                    else:
                        act = preactivation
                    if 'log' in logdir:
                        Wsum.append(tf.summary.histogram(name='Activation'+str(k), values=act))
                        Wsum.append(tf.summary.scalar(name='actsum'+str(k), tensor= tf.reduce_mean(act)))
                    self.h.append(tf.reshape(act, [self.batch_size, -1, act.shape[1].value]))
                    tf.add_to_collection(name('h', k), act)

            try:
                session.run(tf.variables_initializer(params))
            except:
                pass

            regularizer = 0
            count = 0
            with tf.name_scope('Regularizer'):
                for x in params:
                    if 'W' in x.name or 'U' in x.name:
                        regularizer += tf.nn.l2_loss(x)
                        count += tf.size(x)
                regularizer = tf.divide(tf.cast(regularizer, tf.float64), tf.cast(count,tf.float64))
            streaming_reg, streaming_reg_update = tf.contrib.metrics.streaming_mean(regularizer)
            streaming_reg_scalar = tf.summary.scalar('streaming_regularizer', streaming_reg)

            if self.regression:
                with tf.name_scope('Cost'):
                    self.regularizer = regularizer
                    self.mask = tf.sequence_mask(self.length_list)
                    self.flat_predictions = self.h[-1]
                    self.flat_targets = self.targets
                    self.masked_logits = tf.reshape(tf.boolean_mask(self.flat_predictions, self.mask), [-1])
                    self.masked_target = tf.reshape(tf.boolean_mask(self.flat_targets, self.mask), [-1])
                    self.mse = tf.reduce_mean(tf.squared_difference(self.masked_logits, self.masked_target))
                    # REDUCE ALL RELEVANT ERRORS
                    self.cost = tf.reduce_mean(
                         tf.squared_difference(self.masked_logits, self.masked_target)
                        + 0.00005*regularizer)
            else:
                with tf.name_scope('Cost'):
                    # get the last error only
                    batch_range = tf.range(self.batch_size)
                    extended_batch_range = tf.range(tf.multiply(self.batch_size, self.topology[-1]))
                    indices = tf.stack([batch_range, tf.cast(self.length_list - 1, tf.int32)], axis=1)
                    self.last_logits = tf.gather_nd(tf.reshape(self.h[-1], [self.batch_size, -1, self.topology[-1]]), indices)
                    self.simple_target = tf.cast(tf.gather_nd(tf.reshape(self.targets, [self.batch_size, -1]), indices), tf.int32)
                    y_flat = tf.reshape(self.targets, [-1])
                    self.mask = tf.reshape(tf.sequence_mask(self.length_list), [-1])
                    self.masked_logits = tf.boolean_mask(tf.reshape(self.h[-1], [-1, self.topology[-1]]), self.mask)
                    self.masked_logits = tf.reshape(self.masked_logits, [-1, topology[-1]])
                    self.masked_target = tf.cast(tf.boolean_mask(y_flat, self.mask), tf.int32)
                    #creating ids - 0- batchsize*numclasses reproduced for maxlen times
                    self.new_ids = tf.tile(extended_batch_range, [tf.cast(tf.reduce_max(self.length_list), tf.int32)])
                    # transposing the whole thing - it now looks like [ 0, 0, 0 * maxlen ], [1,1,1 ... * maxlen] ,... * batchsize*numclasses
                    self.new_ids_transposed = tf.transpose(tf. reshape(self.new_ids, [-1, tf.multiply(self.batch_size, self.topology[-1])]))
                    # now it gets tricky - turned to shape batchsize, maxlen, numclasses
                    self.new_ids_reshaped = tf.transpose(tf.reshape(self.new_ids_transposed, [self.batch_size, self.topology[-1], -1]), perm=[0,2,1])
                    ### another try
                    #reshaped ids so 00000,11111,22222,....
                    self.flat_ids = tf.reshape(tf.transpose(self.new_ids_reshaped, perm=[0,2,1]),[-1])
                    self.extended_mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.length_list), -1),[1,1, self.topology[-1]])
                    self.reshaped_mask = tf.reshape(tf.transpose(self.extended_mask, perm=[0,2,1]), [-1])
                    self.reshaped_logits = tf.reshape(tf.transpose(self.h[-1], perm=[0,2,1]), [-1])
                    self.masked_flat_ids = tf.boolean_mask(self.flat_ids, self.reshaped_mask)
                    self.masked_flat_logits = tf.boolean_mask(self.reshaped_logits, self.reshaped_mask)
                    self.mean_masked_logits = tf.reshape(tf.segment_mean(self.masked_flat_logits, segment_ids=self.masked_flat_ids), [-1, self.topology[-1]])


                    #REDUCE LAST ERROR
                    #self.cost = tf.reduce_mean(
                    #    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.last_logits, labels=self.simple_target)
                    #    + 0.01 * regularizer)
                    #REDUCE ALL RELEVANT ERRORS
                    #self.cost = tf.reduce_mean(
                    #    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.masked_logits, labels=self.masked_target)
                    #    + 0.01*regularizer)
                    #REDUCE MEAN ERROR
                    self.cost = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.mean_masked_logits, labels=self.simple_target)
                        + 0.01 * regularizer)
            streaming_cost, streaming_cost_update = tf.contrib.metrics.streaming_mean(self.cost)
            streaming_cost_scalar = tf.summary.scalar('streaming_cost', streaming_cost)


        self.mergedperbatch = Wsum
        self.mergedupdate.extend([streaming_cost_update, streaming_reg_update])
        self.mergedperepoch.extend([streaming_cost_scalar, streaming_reg_scalar])
        self.summarywritertrain = tf.summary.FileWriter((logdir + 'train'), self.session.graph)
        self.summarywritertest = tf.summary.FileWriter((logdir + 'test'), self.session.graph)

    def create_optimizer(self):
        '''
        :return:

        Create an Adam optimizer for the column invoking this method - taken outside the init functions
        since not for all columns optimizers are needed at all times. And otherwise it will mess around with some
        parameters.
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.99)

        variables_to_update = self.W + self.b + self.U + self.R
        self.weightUpdates = optimizer.minimize(
            self.cost, var_list=variables_to_update, name=self.name('weight_updates'))
        #they have still all entries! - even those that are not optimized after the end of the utterance
        if self.regression:
            self.predictions = self.h[-1]
            self.confusion_matrix = None
        else:
            self.predictions = tf.argmax(tf.reshape(self.h[-1], [-1, self.topology[-1]]), axis=1, name=self.name('predictions'))

            self.confusion_matrix = tf.confusion_matrix(
                labels=self.simple_target, predictions=tf.argmax(self.last_logits, axis=1), num_classes= self.topology[-1])

            errors = tf.equal(tf.cast(self.predictions, tf.float32), tf.reshape(self.targets, [-1]),
                              name=self.name('errors'))
            #get the last error only
            batch_range = tf.range(self.batch_size)
            indices = tf.stack([batch_range, tf.cast(self.length_list - 1, tf.int32)], axis=1)
            last_errors = tf.gather_nd(tf.reshape(errors, [self.batch_size, -1]), indices)
            self.last_predict = tf.gather_nd(tf.reshape(self.predictions, [self.batch_size, -1]), indices)
            self.simple_target = tf.gather_nd(tf.reshape(self.targets, [self.batch_size, -1]), indices)
            self.mean_predict = tf.argmax(self.mean_masked_logits, axis=1)
            self.mean_errors = tf.equal(self.mean_predict, tf.cast(self.simple_target, tf.int64))

            masked_errors = tf.boolean_mask(errors, tf.reshape(self.mask, [-1]))
            num_samples = tf.size(masked_errors)
            self.last_accuracy = tf.divide(tf.reduce_sum(tf.cast(last_errors, tf.int32)),
                                           self.batch_size, name=self.name('lastacc'))
            self.accuracy = tf.divide(tf.reduce_sum(tf.cast(masked_errors, tf.int32)), num_samples, name=self.name('accuracy'))
            self.mean_acc = tf.divide(tf.reduce_sum(tf.cast(self.mean_errors, tf.int32)),
                                      self.batch_size, name=self.name('meanacc'))
            streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(self.accuracy)
            streaming_accuracy_scalar = tf.summary.scalar('streaming_accuracy', streaming_accuracy)
            self.mergedupdate.append(streaming_accuracy_update)
            self.mergedperepoch.append(streaming_accuracy_scalar)

            streaming_lastaccuracy, streaming_lastaccuracy_update = tf.contrib.metrics.streaming_mean(self.last_accuracy)
            streaming_lastaccuracy_scalar = tf.summary.scalar('streaming_last_accuracy', streaming_lastaccuracy)
            self.mergedupdate.append(streaming_lastaccuracy_update)
            self.mergedperepoch.append(streaming_lastaccuracy_scalar)

        # optimizer_initializers = [var.initializer for var in tf.global_variables() if var not in params]
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        optimizer_initializers = [v.initializer for (v, f) in zip(global_vars, is_not_initialized) if not f]

        try:
            self.session.run(optimizer_initializers)
        except:
            pass
        try:
            self.session.run(tf.local_variables_initializer())
        except:
            pass

        tf.add_to_collection(self.name('weight_updates'), self.weightUpdates)
        tf.add_to_collection(self.name('predictions'), self.predictions)
        return

    def train(self, data, learning_rate: float, dropout_keep_prob = 1.0) ->None:
        '''
        :param data: list of batches to train on containing input, target
        :param learning_rate: float
        :param dropout_keep_prob: float between 0-1
        :return:

        Invokes the current training method for that column.
        '''
        for datapoint in data:
            inputs, targets, lengths = datapoint
            feed_dict={
                self.inputs: inputs,
                self.targets: targets,
                self.learning_rate: learning_rate,
                self.length_list: lengths,
                self.dropout_keep_prob: dropout_keep_prob,
                self.batch_size: len(lengths)
            }

            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
                feed_dict[col.batch_size] = len(lengths)

            self.session.run(self.weightUpdates, feed_dict=feed_dict)

    def write_summary(self, data, step, learning_rate=0, dropout_keep_prob=1, test=False):
        '''
       :param data: Input data
       :param learning_rate: float learning rate
       :param dropout_keep_prob: float dropout keep prob
       :param step: current epoch of the network
       :param test: Bool triggers different data handling and save directory if it is evaluation data

       Trigger the summary writer to write previously constructed summaries
       '''
        self.session.run(tf.local_variables_initializer())
        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths,
                self.batch_size: len(lengths)
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
                feed_dict[col.batch_size] = len(lengths)
            self.session.run(self.mergedupdate, feed_dict=feed_dict)
        summaries = self.session.run(self.mergedperepoch)
        if test:
            for summary in summaries:
                self.summarywritertest.add_summary(summary, step)
        else:
            for summary in summaries:
                self.summarywritertrain.add_summary(summary, step)
        return

    def predict(self, data) -> np.ndarray:
        '''
        :param inputs: list containing the inputs, length and target
        :return: predictions: list with the predictions

        Run a prediction on the input data.
        '''
        predictions = []

        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths,
                self.batch_size: len(lengths)
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
                feed_dict[col.batch_size] = len(lengths)
            if self.regression:
                predictions.extend(
                    self.session.run(self.predictions, feed_dict=feed_dict)
                )
            else:
                predictions.extend(
                    self.session.run(tf.argmax(self.mean_masked_logits, axis=1), feed_dict=feed_dict))
                    #self.session.run(tf.boolean_mask(self.predictions, self.mask), feed_dict=feed_dict))

        return predictions

    def predict_last(self, data) -> np.ndarray:
        '''
        :param inputs: list containing the inputs
        :param lengths: list containing the lengths of the samples
        :return: predictions: list with the predictions

        Run a prediction on the input data for ASC-Data last output.
        '''
        predictions = []
        tars = []

        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
            predic, tar = self.session.run([self.last_predict, self.simple_target], feed_dict=feed_dict)
            predictions.extend(predic)
            tars.extend(tar)

        return predictions, tars

    def get_probabilities(self, data) -> np.ndarray:
        '''
        :param inputs: list containing the inputs
        :param lengths: list containing the lengths of the samples
        :return: probabilities: list with the probabilities

        Run a prediction on the input data and return the logits (ASC-Inclusion only).
        '''
        probabilities = []
        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
            probabilities.extend(
                self.session.run(tf.boolean_mask(tf.nn.softmax(logits=tf.reshape(
                    self.h[-1], [-1, self.topology[-1]])), self.mask), feed_dict= feed_dict))
        return probabilities

    def evaluate(self, data):
        '''
        :param data: dict containing the inputs
        :return: accuracy: The overall accuracy of this prediction (all, last and mean)

        Run a prediction on the input data. Returns only the accuracy (ASC Inclusion only).
        '''
        last = []
        all = []
        mean = []
        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths,
                self.batch_size: len(lengths)
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
                feed_dict[col.batch_size] = len(lengths)
            if self.regression:
                mse = self.session.run([self.cost], feed_dict=feed_dict)
                all.append(mse)
                last.append(0)
                mean.append(0)
            else:
                lastacc, allacc, meanacc = (self.session.run([self.last_accuracy, self.accuracy, self.mean_acc], feed_dict=feed_dict))
                all.append(allacc)
                last.append(lastacc)
                mean.append(meanacc)
        return np.mean(np.asarray(all)), np.mean(np.asarray(last)), np.mean(np.asarray(mean))

    def get_confusion_matrix(self, data):
        '''
        :param data: data as returned from the iterator
        :return: confusion_matrix - the confusion matrix for the full epoch

        Get the confusion matrix for the ASC Inclusion dataset.
        '''
        confusion_matrix = np.zeros((self.topology[-1],self.topology[-1]))
        for inputs, targets, lengths in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
                self.length_list: lengths,
                self.batch_size: len(lengths)
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
                feed_dict[col.length_list] = lengths
                feed_dict[col.batch_size] = len(lengths)
            confusion_matrix += self.session.run(self.confusion_matrix, feed_dict=feed_dict)
        return confusion_matrix


class RNNProgressiveNet(ProgressiveNeuralNetwork):
    '''
        A Progressive Neural Network with an extensible number of columns.
    '''

    def addColumn(self, topology: list, activations: list, celltype,
                       previousColumns: list, logdir: str, batchnum: int, regression=False) -> Column:
        '''
        :param topology: list the numbers of units per layer
        :param activations: list the activations functions of those layers
        :param previousColumns: list a list of the previous columns.
        :param logdir: path to the logging directory
        :param batchnum: batchnumber for the column - needed for initialization
        :param regression: bool indicating if it is a regression or classification task
        :return: e: the just created Column.

        Adding a column to the progressive net.
        '''
        assert self.numLayers == len(activations)
        assert self.numLayers == len(topology)
        topology = [self.inputSize] + topology
        colIndex = len(self.extensibleColumns)
        e = Rnn_Column(
            topology, activations, celltype, self.session, previousColumns, colIndex, logdir=logdir, batchsize= batchnum,
            regression=regression
        )
        self.extensibleColumns.append(e)
        #self.session.run(tf.global_variables_initializer())
        return e

    def loadFromFile(self, filePath: str, finetune= True, file=None) ->None:
        '''
        :param filePath: path
        :param file: NN reload file (if there's a specific instance to load - not the latest)
        :return:

        Loading an existing Prog net from a file.
        '''
        saver = tf.train.Saver()
        if not file == None:
            saver.restore(self.session, file)
        else:
            parentDir = dirname(filePath)
            saver.restore(self.session, tf.train.latest_checkpoint(parentDir))
        return