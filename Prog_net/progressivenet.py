import tensorflow as tf
import numpy as np
from os.path import dirname
import pickle
import os

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



class Column:
    '''
    An extensible network column for use in transfer learning with a Progressive
    Neural Network.
    '''

    def __init__(self, topology: list, activations: list,
                 session, previousColumns: list, colIndex: int, dtype=tf.float64, logdir='',
                 regression=False
    ) -> None:
        '''
        :param topology: list containing the number of units per layer.
        :param activations: list containing the activation functions of those layers.
        :param session: tensorflow session to connect with this column
        :param previousColumns: list containing the previous columns of the prog net.
        :param colIndex: int index of the current column - used for naming
        :param dtype: Standard tf datatype
        :param logdir: directory to write logs and saved files to.

        Creating one column for a Progressive neural net.
        '''

        def name(*args, **kwargs) -> str:
            return  self.name(*args, **kwargs)

        n_input = topology[0]
        self.colIndex = colIndex
        self.topology = topology
        self.session = session
        self.merged = None
        self.summarywritertrain = None
        self.summarywritertest = None
        self.confusion_matrix = None

        width = len(previousColumns)

        # numLayers in network - first value doesn't count.

        numLayers = self.numLayers = len(topology) - 1
        prevCols = self.previousColumns = previousColumns

        #Columns must have the same height
        assert all([self.numLayers == col.numLayers for col in prevCols])
        with tf.name_scope(str(colIndex)):

            with tf.name_scope('inputs'):
                self.inputs = tf.placeholder(
                    dtype, shape=[None, n_input], name=name('inputs')
                )
            with tf.name_scope('targets'):
                self.targets = tf.placeholder(
                    dtype, shape=[None, topology[-1]], name=name('targets')
                )
            with tf.name_scope('learning_rate'):
                self.learning_rate = tf.placeholder_with_default(
                    np.float64(0.0), shape=(), name=name('learning_rate')
                )
            learnsum = tf.summary.scalar('learning_rate', self.learning_rate)
            with tf.name_scope('dropout'):
                self.dropout_keep_prob = tf.placeholder_with_default(
                    np.float64(1.0), shape=(), name=name('dropout_keep_prob')
                )
            dropsum = tf.summary.scalar('dropout', self.dropout_keep_prob)

            tf.add_to_collection(name('dropout_keep_prob'), self.dropout_keep_prob)
            tf.add_to_collection(name('learning_rate'), self.learning_rate)

            self.W = [[]] * numLayers
            self.b = [[]] * numLayers
            self.U = []
            for k in range(numLayers-1):
                self.U.append( [[]] * width )
            self.h = [self.inputs]
            params = []
            Wsum =[]

            for k in range(numLayers):
                with tf.name_scope(str(k)):
                    W_shape = topology[k:k+2]
                    Wk = self.W[k] = weight_variable(W_shape, name('W', k))
                    bk = self.b[k] = bias_variable([W_shape[1]], name('b', k))
                    if 'log' in logdir:
                        Wsum.append(tf.summary.histogram(name='Weights'+str(k), values=Wk))

                    with tf.name_scope('Preactivation'):
                        preactivation = tf.matmul(self.h[-1], Wk) + bk
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
                                preactivation += tf.matmul(prevCols[kk].h[k], Uk[kk])
                        for kk in range(width):
                            params.append(self.U[k-1][kk])
                    with tf.name_scope('Activations'):
                        act = activations[k](preactivation)
                    if k<= numLayers-1:
                        with tf.name_scope('Dropouts'):
                            act = tf.nn.dropout(act, self.dropout_keep_prob)
                    if 'log' in logdir:
                        Wsum.append(tf.summary.histogram(name='Activation'+str(k), values=act))
                        Wsum.append(tf.summary.scalar(name='actsum'+str(k), tensor= tf.reduce_mean(act)))
                    self.h.append(act)
                    tf.add_to_collection(name('h', k), act)

            try:
                session.run(tf.variables_initializer(params))
            except:
                pass

            regularizer = 0
            with tf.name_scope('Regularizer'):
                for x in params:
                    if 'W' or 'U' in x.name:
                        regularizer += tf.nn.l2_loss(x)
            regsum = tf.summary.scalar('Regularizer', regularizer)
            if regression:
                with tf.name_scope('Cost'):
                    self.cost = tf.reduce_mean(tf.nn.l2_loss(self.h[-1] - self.targets) + 0.01 * regularizer)
            else:
                with tf.name_scope('Cost'):
                    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels= self.targets, logits= self.h[-1]) + 0.01 * regularizer)
            costsum = tf.summary.scalar('cost', self.cost)

            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.merged = tf.summary.merge((regsum, costsum, learnsum, dropsum, Wsum))
        self.summarywritertrain = tf.summary.FileWriter((logdir+'train'), self.session.graph)
        self.summarywritertest = tf.summary.FileWriter((logdir + 'test'), self.session.graph)
        return


    def write_summary(self, data, learning_rate, dropout_keep_prob, step=0, test=False):
        '''
        :param data: Input data
        :param learning_rate: float learning rate
        :param dropout_keep_prob: float dropout keep prob
        :param step: current epoch of the network
        :param test: Bool triggers different data handling and save directory if it is evaluation data

        Trigger the summary writer to write previously constructed summaries
        '''
        if test:
            inputs = data[0]
            targets = data[1]
        else:
            inputs = []
            targets = []
            for item1, item2 in data:
                inputs.extend(item1)
                targets.extend(item2)
        feed_dict={
            self.inputs: inputs,
            self.targets: targets,
            self.learning_rate: learning_rate,
            self.dropout_keep_prob: dropout_keep_prob
        }

        for col in self.previousColumns:
            feed_dict[col.inputs] = inputs
        summary = self.session.run(self.merged, feed_dict = feed_dict)
        if test:
            self.summarywritertest.add_summary(summary, step)
        else:
            self.summarywritertrain.add_summary(summary, step)

    def create_summary_from_array(self, data, steps, tag= '', test=True):
        '''
        :param data: list the array to create the summary from
        :param steps: list with the time steps of the datapoints
        :param tag: string name of the datapoint
        :param test: bool indicating wether it is test or train data for the directory
        :return:

        Create a tensorflow summary from an array.
        '''
        try:
            for counter, point in enumerate(data):
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag=tag, simple_value=point),
                ])
                if test:
                    self.summarywritertest.add_summary(summary, steps[counter])
                else:
                    self.summarywritertrain.add_summary(summary, steps[counter])
        except TypeError:
            return
        return

    def create_optimizer(self, dataset=None):
        '''
        :param dataset: string indicating if it is the deenigma dataset and therefore needs a regressor
        :return:

        Create an Adam optimizer for the column invoking this method - taken outside the init functions
        since not for all columns optimizers are needed at all times. And otherwise it will mess around with some
        parameters.
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2= 0.99)


        variables_to_update = self.W + self.b + self.U
        self.weightUpdates = optimizer.minimize(
                self.cost, var_list=variables_to_update, name=self.name('weight_updates'))

        if dataset == 'deenigma':
            self.predictions = self.h[-1]
            self.accuracy = tf.reduce_mean(self.h[-1] - self.targets)

        else:
            self.predictions = tf.argmax(self.h[-1], axis=1, name=self.name('predictions'))


            errors = tf.equal(tf.cast(self.predictions, tf.int64), tf.argmax(self.targets, axis=-1), name=self.name('errors'))
            num_samples = tf.size(self.predictions)
            self.accuracy = tf.divide(tf.reduce_sum(tf.cast(errors, tf.int32)), num_samples, name=self.name('accuracy'))
            acsum = tf.summary.scalar('Accuracy', self.accuracy)
            self.merged = tf.summary.merge((self.merged, acsum))

            self.confusion_matrix = tf.confusion_matrix(
                labels=tf.argmax(self.targets, axis=-1), predictions=self.predictions, num_classes=self.topology[-1])

        #optimizer_initializers = [var.initializer for var in tf.global_variables() if var not in params]
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        optimizer_initializers = [v.initializer for (v, f) in zip(global_vars, is_not_initialized) if not f]

        try:
            self.session.run(optimizer_initializers)
        except:
            pass

        tf.add_to_collection(self.name('weight_updates'), self.weightUpdates)
        tf.add_to_collection(self.name('predictions'), self.predictions)

    def create_optimizer_finetune(self, lastlayeronly = True, graph = None):
        '''
        :param lastlayeronly: Bool if true only the last parameters are updated
        :param graph: tensorflow graph currently dealt with
        :return:

        Create an Adam finetuning optimizer for the column invoking this method - taken outside the init functions
            since not for all columns optimizers are needed at all times. And otherwise it will mess around with some
            parameters.
        '''

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.99, name='fine')
        if lastlayeronly:
            variables_to_update = [self.W[-1]] + [self.b[-1]] + [self.U[-1]]
        else:
            variables_to_update = self.W + self.b + self.U
        self.weightUpdates = optimizer.minimize(
            self.cost, var_list=variables_to_update, name=self.name('weight_updates'))
        self.predictions = tf.argmax(self.h[-1], axis=1, name=self.name('predictions'))

        errors = tf.equal(tf.cast(self.predictions, tf.int64), tf.argmax(self.targets, axis=-1),
                          name=self.name('errors'))
        num_samples = tf.size(self.predictions)
        self.accuracy = tf.divide(tf.reduce_sum(tf.cast(errors, tf.int32)), num_samples, name=self.name('accuracy'))
        acsum = tf.summary.scalar('Accuracy', self.accuracy)
        self.merged = tf.summary.merge((self.merged, acsum))
        self.confusion_matrix = tf.confusion_matrix(
            labels=tf.argmax(self.targets, axis=-1), predictions=self.predictions, num_classes=self.topology[-1])



        #optimizer_initializers = [var.initializer for var in tf.global_variables() if 'fine' in var.name]
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        optimizer_initializers = [v.initializer for (v, f) in zip(global_vars, is_not_initialized) if not f]

        try:
            self.session.run(optimizer_initializers)
        except:
            pass

        tf.add_to_collection(self.name('weight_updates'), self.weightUpdates)
        tf.add_to_collection(self.name('predictions'), self.predictions)

    def name(self, name: str, index1 = None, index2 = None) -> str:
        '''
        :param name: str name of the current variable to name.
        :param index1: int first index
        :param index2: int second index - if it applies
        :return: str a string with the correct name.

        Helper function creating a name for the current variable.
        '''
        if index1 is None and index2 is None:
            return 'Column{}_{}'.format(self.colIndex, name)
        elif index1 is not None and index2 is None:
            tmp = 'Column{}_{}_{}'
            return tmp.format(self.colIndex, name, index1)
        else:
            tmp = 'Column{}_{}_{}_{}'
            return  tmp.format(self.colIndex, name, index1, index2)

    def train(self, data, learning_rate: float, dropout_keep_prob = 1.0) ->None:
        '''
        :param data: list of batches to train on containing input, target
        :param learning_rate: float learningrate
        :param dropout_keep_prob: float between 0-1
        :return:

        Invokes the current training method for that column.
        '''
        for datapoint in data:
            inputs, targets = datapoint
            feed_dict={
                self.inputs: inputs,
                self.targets: targets,
                self.learning_rate: learning_rate,
                self.dropout_keep_prob: dropout_keep_prob
            }

            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs

            self.session.run(self.weightUpdates, feed_dict=feed_dict)

    def predict(self, inputs) -> np.ndarray:
        '''
        :param inputs: list containing the inputs
        :return: predictions: list with the predictions

        Run a prediction on the input data.
        '''
        feed_dict = {self.inputs: inputs}
        for col in self.previousColumns:
            feed_dict[col.inputs] = inputs
        return self.session.run(self.predictions, feed_dict=feed_dict)

    def evaluate(self, data, test=True):
        '''
        :param data: dict containing the inputs
        :param test: bool indicating evaluation data
        :return: accuracy: The overall accuracy of this prediction

        Run a prediction on the input data. Returns only the accuracy.
        '''
        if test:
            inputs = data['data']
            targets = data['target']
        else:
            inputs = []
            targets = []
            for item1, item2 in data:
                inputs.extend(item1)
                targets.extend(item2)
        feed_dict = {
            self.inputs: inputs,
            self.targets: targets
        }
        for col in self.previousColumns:
            feed_dict[col.inputs] = inputs
        accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        return accuracy

    def get_confusion_matrix(self, data) -> None:
        '''
        :param data: data as returned from the iterator
        :return: confusion_matrix - the confusion matrix for the full epoch

        Get the confusion matrix for ASC Data.
        '''
        confusion_matrix = np.zeros((self.topology[-1], self.topology[-1]))
        for inputs, targets in data:
            feed_dict = {
                self.inputs: inputs,
                self.targets: targets,
            }
            for col in self.previousColumns:
                feed_dict[col.inputs] = inputs
            confusion_matrix += self.session.run(self.confusion_matrix, feed_dict=feed_dict)
        return confusion_matrix



class ProgressiveNeuralNetwork:
    '''
    A Progressive Neural Network with an extensible number of columns.
    '''

    def __init__(self, inputSize: int, numLayers: int) -> None:
        '''
        :param inputSize: setting the size of the input.
        :param numLayers: Number of layers of the prog net.

        Initializing the network with some basic parameters.
        '''
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.session = session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.initialColumn = None
        self.extensibleColumns = []
        self.graph = None
        self.saverprot = None


    def addColumn(self, topology: list, activations: list,
                       previousColumns: list, logdir: str, regression:bool) -> Column:
        '''
        :param topology: list the numbers of units per layer
        :param activations: list the activations functions of those layers
        :param previousColumns: list a list of the previous columns.
        :param logdir: string directory to save data about this column
        :param regression: boolean indicating if this is a regression column
        :return: e: the just created Column.

        Adding a column to the progressive net.
        '''

        if regression == None:
            regression=False
        assert self.numLayers == len(activations)
        assert self.numLayers == len(topology)
        topology = [self.inputSize] + topology
        colIndex = len(self.extensibleColumns)
        e = Column(
            topology, activations, self.session, previousColumns, colIndex, logdir=logdir, regression=regression
        )
        self.extensibleColumns.append(e)
        #self.session.run(tf.global_variables_initializer())
        return e

    def loadFromFile(self, filePath: str) ->None:
        '''
        :param filePath: path
        :return:

        Loading an existing Prog net from a file.
        '''

        saver = tf.train.Saver()
        parentDir = dirname(filePath)
        saver.restore(self.session, tf.train.latest_checkpoint(parentDir))

        return

    def writeToFile(self, filePath: str, step: int, meta = True, trainable= False) -> None:
        '''
        :param filePath: str path to save it to
        :param step: int current global step
        :param meta: bool indicating if meta information (the graph) should be saved or only params
        :param trainable: save trainable variables only
        :return:

        Writing the current Prog net to a file.
        '''
        #var_list = [v for v in tf.global_variables() if "adam" not in v.name]
        #saver = tf.train.Saver(var_list)
        if trainable == True:
            saver = tf.train.Saver(tf.trainable_variables(), saver_def=self.saverprot)
        else:
            saver = tf.train.Saver(saver_def=self.saverprot)
        saver.save(self.session, filePath, global_step=step, write_meta_graph=meta)

    def limit_mem(self):
        '''
        :return:

        A function to close the session and release memory.
        '''
        self.session.close()
        tf.reset_default_graph()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        #self.session= tf.Session(config=cfg)
