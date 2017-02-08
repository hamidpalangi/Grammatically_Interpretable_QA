import tensorflow as tf

class TPRCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, nSymbols, nRoles, dSymbols, dRoles, activation=tf.nn.sigmoid):
        """
        Tensor Product Representation (TPR) Cell
        T = F * B * R^T where the dimensions are as follow:
        F: dSymbols x nSymbols
        B: nSymbols x nRoles
        R: dRoles x nRoles
        T: dSymbols x dRoles

        B = a^F * (a^R)^T where the dimensions are:
        a^F: nSymbols x 1
        a^R: nRoles x 1

        :param nSymbols: # of symbols
        :param nRoles: # of roles
        :param dSymbols: embedding size of each symbol
        :param dRoles: embedding size of each role
        :param activation: non-linear activation function
        """
        self._nSymbols = nSymbols
        self._nRoles = nRoles
        self._dSymbols = dSymbols
        self._dRoles = dRoles
        self._activation = activation

    @property
    def state_size(self):
        return self._dSymbols * self._dRoles

    @property
    def output_size(self):
        return self._dSymbols * self._dRoles

    def __call__(self, inputs, state, scope=None):
        """
        :param inputs:
        :param state: is basically vec(T).
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("BindVecs_aF"):
                # Dimensionality of aF will be [batchsize x nSymbols].
                aF = self._activation(tf.nn.rnn_cell._linear([inputs, state], output_size=self._nSymbols, bias=False))
            with tf.variable_scope("BindVecs_aR"):
                # Dimensionality of aR will be [batchsize x nRoles].
                aR = self._activation( tf.nn.rnn_cell._linear([inputs, state], output_size=self._nRoles, bias=False) )
            with tf.variable_scope("FillerRoles"):
                F = tf.get_variable(name="F", shape=[self._nSymbols, self._dSymbols])
                R = tf.get_variable(name="R", shape=[self._nRoles, self._dRoles])
                # Dimensionality of itemF will be [batchsize x dSymboles]
                # Dimensionality of itemR will be [batchsize x dRoles]
                itemF = tf.matmul(aF, F)
                itemR = tf.matmul(aR, R)
                # Preparing itemF and itemR to use them with batch_matmul. E.g., the new dimension for itemF
                # will be [batchsize x dSymbols x 1]. The new dimension of itemR will be [batchsize x dRoles x 1].
                # Note that each slice is transposed in "itemR" during "batch_matmul".
                itemF = tf.expand_dims(itemF, 2)
                itemR = tf.expand_dims(itemR, 2)
            T = tf.batch_matmul( x = itemF, y = itemR, adj_y=True)
            # Vectorizing T. The dimension of new_state will be [batchsize x (dSymbols*dRoles)]
            new_state = tf.reshape(T, shape=[tf.shape(T)[0], -1])
        return new_state, new_state
