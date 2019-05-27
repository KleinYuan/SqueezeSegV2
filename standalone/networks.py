import numpy as np
import tensorflow as tf


class SqueezeSegV2(object):
    """
    Modified SqueezeSegV2 feature extractor
    """

    def forward(self, X, feature_dim, keep_prob, is_training):

        """NN architecture."""

        _conv1 = self._conv_bn_layer(
            X, 'conv1', 'bias', 'scale',
            filters=64, size=3, stride=2,
            padding='SAME', freeze=False,
            conv_with_bias=True, stddev=0.001, is_training=is_training)

        _ca1 = self._context_aggregation_layer('se1', _conv1)
        _conv1_skip = self._conv_bn_layer(
            X, 'conv1_skip', 'bias', 'scale',
            filters=64, size=1, stride=1, padding='SAME', freeze=False,
            conv_with_bias=True, stddev=0.001, is_training=is_training)
        _pool1 = self._pooling_layer('pool1', _ca1, size=3, stride=2, padding='SAME')

        _fire2 = self._fire_layer('fire2', _pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False, is_training=is_training)
        _ca2 = self._context_aggregation_layer('se2', _fire2)

        _fire3 = self._fire_layer('fire3', _ca2, s1x1=16, e1x1=64, e3x3=64, freeze=False, is_training=is_training)
        _ca3 = self._context_aggregation_layer('se3', _fire3)
        _pool3 = self._pooling_layer('pool3', _ca3, size=3, stride=2, padding='SAME')

        _fire4 = self._fire_layer('fire4', _pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False, is_training=is_training)
        _fire5 = self._fire_layer('fire5', _fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False, is_training=is_training)
        _pool5 = self._pooling_layer('pool5', _fire5, size=3, stride=2, padding='SAME')

        _fire6 = self._fire_layer('fire6', _pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False, is_training=is_training)
        _fire7 = self._fire_layer('fire7', _fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False, is_training=is_training)
        _fire8 = self._fire_layer('fire8', _fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False, is_training=is_training)
        _fire9 = self._fire_layer('fire9', _fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False, is_training=is_training)

        _fire10 = self._fire_deconv('fire_deconv10', _fire9, s1x1=64, e1x1=128, e3x3=128, factors=[1, 2], stddev=0.1, is_training=is_training)
        _fire10_fuse = tf.add(_fire10, _fire5, name='fure10_fuse')

        _fire11 = self._fire_deconv('fire_deconv11', _fire10_fuse, s1x1=32, e1x1=64, e3x3=64, factors=[1, 2], stddev=0.1, is_training=is_training)
        _fire11_fuse = tf.add(_fire11, _ca3, name='fire11_fuse')

        _fire12 = self._fire_deconv('fire_deconv12', _fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2], stddev=0.1, is_training=is_training)
        _fire12_fuse = tf.add(_fire12, _ca1, name='fire12_fuse')

        _fire13 = self._fire_deconv('fire_deconv13', _fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2], stddev=0.1, is_training=is_training)
        _fire13_fuse = tf.add(_fire13, _conv1_skip, name='fire13_fuse')

        _drop13 = tf.layers.dropout(_fire13_fuse, keep_prob, training=is_training, name='drop13')
        feature = self._conv_layer('conv14_prob', _drop13, filters=feature_dim, size=3, stride=1, padding='SAME', relu=False, stddev=0.1, init=True)

        return feature

    def _variable_on_device(self, name, shape, initializer, trainable=True, dtype=tf.float32):
        """Helper to create a Variable.

          Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

          Returns:
            Variable Tensor
        """
        if not callable(initializer):
            var = tf.get_variable(name, initializer=initializer, trainable=trainable)
        else:
            var = tf.get_variable(
                name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, wd, initializer, trainable=True):
        """Helper to create an initialized Variable with weight decay.

              Note that the Variable is initialized with a truncated normal distribution.
              A weight decay is added only if one is specified.

              Args:
                name: name of the variable
                shape: list of ints
                wd: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.

              Returns:
                Variable Tensor
          """
        var = self._variable_on_device(name, shape, initializer, trainable)
        if wd is not None and trainable:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, is_training, stddev=0.001,
                    freeze=False):
        """Fire layer constructor.
        Args:
          layer_name: layer name
          inputs: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          freeze: if true, do not train parameters in this layer.
        Returns:
          fire layer operation.
        """
        sq1x1 = self._conv_bn_layer(
            inputs, layer_name + '/squeeze1x1', 'bias', 'scale',
            filters=s1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        ex1x1 = self._conv_bn_layer(
            sq1x1, layer_name + '/expand1x1', 'bias', 'scale',
            filters=e1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        ex3x3 = self._conv_bn_layer(
            sq1x1, layer_name + '/expand3x3', 'bias', 'scale',
            filters=e3x3, size=3, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        return tf.concat([ex1x1, ex3x3], 3, name=layer_name + '/concat')

    def _fire_deconv(self, layer_name, inputs, s1x1, e1x1, e3x3, is_training,
                     factors=[1, 2], freeze=False, stddev=0.001):
        """Fire deconvolution layer constructor.
        Args:
          layer_name: layer name
          inputs: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          factors: spatial upsampling factors.
          freeze: if true, do not train parameters in this layer.
        Returns:
          fire layer operation.
        """
        assert len(factors) == 2, 'factors should be an array of size 2'

        ksize_h = factors[0] * 2 - factors[0] % 2
        ksize_w = factors[1] * 2 - factors[1] % 2
        sq1x1 = self._conv_bn_layer(
            inputs, layer_name + '/squeeze1x1', 'bias', 'scale',
            filters=s1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        deconv = self._deconv_layer(
            layer_name + '/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
            stride=factors, padding='SAME', init='bilinear')
        ex1x1 = self._conv_bn_layer(
            deconv, layer_name + '/expand1x1', 'bias', 'scale',
            filters=e1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        ex3x3 = self._conv_bn_layer(
            deconv, layer_name + '/expand3x3', 'bias', 'scale',
            filters=e3x3, size=3, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev, is_training=is_training)
        return tf.concat([ex1x1, ex3x3], 3, name=layer_name + '/concat')

    def _context_aggregation_layer(self, layer_name, inputs, REDUCTION=16):
        """
    The maxpooling layer  followed by two cascaded convolution layers with
    a ReLU activation in between. Use the sigmoid function to normalize the
    output of the module and use an element-wise multiplication to combine
    the output with the input.

    Args:
      layer_name: layer name.
      inputs: input tensor
    Returns:
      outputs: tensor with same shape as input tensor.
    """
        with tf.variable_scope(layer_name) as scope:
            pool = self._pooling_layer(
                'pool', inputs, size=7, stride=1, padding='SAME')
            pool_shape = pool.get_shape().as_list()
            pool_dim = pool_shape[-1]
            squeeze = self._conv_layer(
                'squeeze', pool, filters=(pool_dim // REDUCTION), size=1, stride=1,
                padding='SAME', freeze=False, xavier=True, relu=True)
            excitation = self._conv_layer(
                'excitation', squeeze, filters=pool_dim, size=1, stride=1,
                padding='SAME', freeze=False, xavier=True, relu=False)
            excitation = tf.nn.sigmoid(excitation)
            scale = inputs * excitation

            return scale

    def _conv_bn_layer(
            self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
            size, stride, is_training, padding='SAME', freeze=False, relu=True,
            conv_with_bias=True, stddev=0.001, decay=0.999, WEIGHT_DECAY=0.0005, BATCH_NORM_EPSILON=1e-5):
        """ Convolution + [relu] + BatchNorm layer. Batch mean and var are treated
            as constant. Weights have to be initialized from a pre-trained model or
            restored from a checkpoint.
            Args:
              inputs: input tensor
              conv_param_name: name of the convolution parameters
              bn_param_name: name of the batch normalization parameters
              scale_param_name: name of the scale parameters
              filters: number of output filters.
              size: kernel size.
              stride: stride
              padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
              freeze: if true, then do not train the parameters in this layer.
              xavier: whether to use xavier weight initializer or not.
              relu: whether to use relu or not.
              conv_with_bias: whether or not add bias term to the convolution output.
              stddev: standard deviation used for random weight initializer.
            Returns:
              A convolutional layer operation.
        """
        with tf.variable_scope(conv_param_name) as scope:
            channels = inputs.get_shape()[3]

            kernel_init = tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float32)
            if conv_with_bias:
                bias_init = tf.constant_initializer(0.0)
            gamma_val = tf.constant_initializer(1.0)
            beta_val = tf.constant_initializer(0.0)
            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            kernel = self._variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            if conv_with_bias:
                biases = self._variable_on_device('biases', [filters], bias_init, trainable=(not freeze))

            gamma = self._variable_on_device('gamma', [filters], gamma_val, trainable=(not freeze))
            beta = self._variable_on_device('beta', [filters], beta_val, trainable=(not freeze))
            moving_mean = self._variable_on_device('moving_mean', [1, 1, 1, filters], tf.zeros_initializer, trainable=False)
            moving_var = self._variable_on_device('moving_var', [1, 1, 1, filters], tf.ones_initializer, trainable=False)

            conv = tf.nn.conv2d(
                inputs, kernel, [1, 1, stride, 1], padding=padding,
                name='convolution')
            if conv_with_bias:
                conv = tf.nn.bias_add(conv, biases, name='bias_add')

            out_shape = conv.get_shape().as_list()

            num_flops = \
                (1 + 2 * int(channels) * size * size) * filters * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]

            if relu:
                conv = tf.nn.relu(conv)

            # Batch Normalization
            mean, var = tf.nn.moments(conv, [0, 1, 2], name='moments')

            # During training, using batchwise mean and var
            updated_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
            updated_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
            with tf.control_dependencies([updated_mean, updated_var]):
                conv_train = tf.nn.batch_normalization(conv, mean=mean, variance=var,
                                                       offset=beta, scale=gamma,
                                                       variance_epsilon=BATCH_NORM_EPSILON, name='batch_norm_train')

            # Testing using moving mean and moving var
            conv_test = tf.nn.batch_normalization(conv, mean=moving_mean, variance=moving_var,
                                                  offset=beta, scale=gamma,
                                                  variance_epsilon=BATCH_NORM_EPSILON, name='batch_norm_test')
            conv = tf.cond(is_training, lambda: conv_train, lambda: conv_test)
            # if is_training:
            #     # During training, using batchwise mean and var
            #     updated_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
            #     updated_var = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
            #     with tf.control_dependencies([updated_mean, updated_var]):
            #         conv = tf.nn.batch_normalization(
            #             conv, mean=mean, variance=var, offset=beta, scale=gamma,
            #             variance_epsilon=BATCH_NORM_EPSILON, name='batch_norm')
            # else:
            #     # Testing using moving mean and moving var
            #     conv = tf.nn.batch_normalization(
            #         conv, mean=moving_mean, variance=moving_var, offset=beta, scale=gamma,
            #         variance_epsilon=BATCH_NORM_EPSILON, name='batch_norm')

            return conv

    def _conv_layer(
            self, layer_name, inputs, filters, size, stride, padding='SAME',
            freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0, init=False, WEIGHT_DECAY=0.0005):
        """Convolutional layer operation constructor.

            Args:
              layer_name: layer name.
              inputs: input tensor
              filters: number of output filters.
              size: kernel size.
              stride: stride
              padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
              freeze: if true, then do not train the parameters in this layer.
              xavier: whether to use xavier weight initializer or not.
              relu: whether to use relu or not.
              stddev: standard deviation used for random weight initializer.
            Returns:
                A convolutional layer operation.
        """
        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]

            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(bias_init_val)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(bias_init_val)

            if init:
                bias_init_val = np.array([0.01, 0.33, 0.33, 0.33])
                bias_init = tf.constant_initializer(-np.log((1 - bias_init_val) / bias_init_val))
            kernel = self._variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            biases = self._variable_on_device('biases', [filters], bias_init,
                                         trainable=(not freeze))

            conv = tf.nn.conv2d(
                inputs, kernel, [1, 1, stride, 1], padding=padding,
                name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            return out

    def _deconv_layer(
            self, layer_name, inputs, filters, size, stride, padding='SAME',
            freeze=False, init='trunc_norm', relu=True, stddev=0.001, WEIGHT_DECAY=0.0005):
        """
            Deconvolutional layer operation constructor.

            Args:
              layer_name: layer name.
              inputs: input tensor
              filters: number of output filters.
              size: kernel size. An array of size 2 or 1.
              stride: stride. An array of size 2 or 1.
              padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
              freeze: if true, then do not train the parameters in this layer.
              init: how to initialize kernel weights. Now accept 'xavier',
                  'trunc_norm', 'bilinear'
              relu: whether to use relu or not.
              stddev: standard deviation used for random weight initializer.
            Returns:
              A convolutional layer operation.
        """

        assert len(size) == 1 or len(size) == 2, \
            'size should be a scalar or an array of size 2.'
        assert len(stride) == 1 or len(stride) == 2, \
            'stride should be a scalar or an array of size 2.'
        assert init == 'xavier' or init == 'bilinear' or init == 'trunc_norm', \
            'initi mode not supported {}'.format(init)

        if len(size) == 1:
            size_h, size_w = size[0], size[0]
        else:
            size_h, size_w = size[0], size[1]

        if len(stride) == 1:
            stride_h, stride_w = stride[0], stride[0]
        else:
            stride_h, stride_w = stride[0], stride[1]

        with tf.variable_scope(layer_name) as scope:
            _, in_height, in_width, channels = inputs.get_shape().as_list()
            batch_size = tf.shape(inputs)[0]

            if init == 'xavier':
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            elif init == 'bilinear':
                assert size_h == 1, 'Now only support size_h=1'
                assert channels == filters, \
                    'In bilinear interporlation mode, input channel size and output' \
                    'filter size should be the same'
                assert stride_h == 1, \
                    'In bilinear interpolation mode, stride_h should be 1'

                kernel_init = np.zeros(
                    (size_h, size_w, channels, channels),
                    dtype=np.float32)

                factor_w = (size_w + 1) // 2
                assert factor_w == stride_w, \
                    'In bilinear interpolation mode, stride_w == factor_w'

                center_w = (factor_w - 1) if (size_w % 2 == 1) else (factor_w - 0.5)
                og_w = np.reshape(np.arange(size_w), (size_h, -1))
                up_kernel = (1 - np.abs(og_w - center_w) / factor_w)
                for c in xrange(channels):
                    kernel_init[:, :, c, c] = up_kernel

                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            # Kernel layout for deconv layer: [H_f, W_f, O_c, I_c] where I_c is the
            # input channel size. It should be the same as the channel size of the
            # input tensor.
            kernel = self._variable_with_weight_decay(
                'kernels', shape=[size_h, size_w, filters, channels],
                wd=WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))
            biases = self._variable_on_device(
                'biases', [filters], bias_init, trainable=(not freeze))

            deconv = tf.nn.conv2d_transpose(
                inputs, kernel,
                [batch_size, stride_h * in_height, stride_w * in_width, filters],
                [1, stride_h, stride_w, 1], padding=padding,
                name='deconv')
            deconv_bias = tf.nn.bias_add(deconv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(deconv_bias, 'relu')
            else:
                out = deconv_bias

            out_shape = out.get_shape().as_list()
            num_flops = \
                (1 + 2 * channels * size_h * size_w) * filters * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]

            return out

    def _pooling_layer(self, layer_name, inputs, size, stride, padding='SAME'):
        """Pooling layer operation constructor.
            Args:
              layer_name: layer name.
              inputs: input tensor
              size: kernel size.
              stride: stride
              padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
            Returns:
              A pooling layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            out = tf.nn.max_pool(inputs,
                                 ksize=[1, size, size, 1],
                                 strides=[1, 1, stride, 1],
                                 padding=padding)
            return out

    def _fc_layer(self, layer_name, inputs, hiddens, flatten=False, relu=True,
            xavier=True, stddev=0.001, bias_init_val=0.0, weight_decay=0.0005):
        """Fully connected layer operation constructor.

            Args:
              layer_name: layer name.
              inputs: input tensor
              hiddens: number of (hidden) neurons in this layer.
              flatten: if true, reshape the input 4D tensor of shape
                  (batch, height, weight, channel) into a 2D tensor with shape
                  (batch, -1). This is used when the input to the fully connected layer
                  is output of a convolutional layer.
              relu: whether to use relu or not.
              xavier: whether to use xavier weight initializer or not.
              stddev: standard deviation used for random weight initializer.
            Returns:
              A fully connected layer operation.
        """
        with tf.variable_scope(layer_name) as scope:
            input_shape = inputs.get_shape().as_list()
            if flatten:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                inputs = tf.reshape(inputs, [-1, dim])

            else:
                dim = input_shape[1]

            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(bias_init_val)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(bias_init_val)

            weights = self._variable_with_weight_decay(
                'weights', shape=[dim, hiddens], wd=weight_decay,
                initializer=kernel_init)
            biases = self._variable_on_device('biases', [hiddens], bias_init)

            outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            if relu:
                outputs = tf.nn.relu(outputs, 'relu')

            num_flops = 2 * dim * hiddens + hiddens
            if relu:
                num_flops += 2 * hiddens

            return outputs


def test():
    print("Testing ........")
    import cv2

    batch_size = 1
    B, H, W, C = (batch_size, 300, 1200, 1)

    test_networks = SqueezeSegV2()

    with tf.variable_scope('Placeholders'):
        x_dm = tf.placeholder(tf.float32, [None, H, W, C])
        is_training = tf.placeholder(tf.bool)

    x_dm_fed = np.random.rand(B, H, W, C).astype(np.float32)

    feed_dict = {
        x_dm: x_dm_fed,
        is_training: True
    }

    z_dm = test_networks.forward(X=x_dm, feature_dim=1024, keep_prob=0.5, is_training=is_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        z_dm_value = sess.run(z_dm, feed_dict=feed_dict)
        print("Out Put shape is {}".format(z_dm_value.shape))
        assert z_dm_value.shape == (B, H, W, 1024)
    cv_in = x_dm_fed[0]*255
    cv_out = z_dm_value[0][:, :, [0]]*255
    overlay = np.concatenate([cv_in, cv_out], 0)
    cv2.imshow('overlay', overlay.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
