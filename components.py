# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

import tflearn
from tflearn import variables as vs
from tflearn import activations
from tflearn import initializations
from tflearn import losses
from tflearn import utils

def condition(cond, t, f):
    if cond is True:
        return t
    elif cond is False:
        return f
    else:
        return tf.cond(cond, lambda: t, lambda: f)

class objectview(object):
    def __init__(self, d):
        self.__dict__.update(d)

componentInherit = {
    'globalDroppath': False,
    'localDroppath': False,
    'localDroppathProb': .5,
    'parentType': '',
    'currentType': ''
}
class TFComponent:
    def __getitem__(self, incoming):
        global componentInherit
        inheritBak = componentInherit.copy()
        if 'localDroppath' in self.opts:
            componentInherit['localDroppath'] = self.opts['localDroppath']
        if 'globalDroppath' in self.opts:
            componentInherit['globalDroppath'] = self.opts['globalDroppath']
        componentInherit['parentType'] = componentInherit['currentType']
        componentInherit['currentType'] = type(self).__name__
        opts = objectview(self.opts)
        if isinstance(incoming, TFComponentVal) and (not hasattr(self, 'noDirect')):
            incoming = incoming.resolve()
        net = self.get(incoming, opts, componentInherit)
        if isinstance(net, TFComponentVal) and componentInherit['parentType'] is '':
            net = net.resolve()
        componentInherit = inheritBak
        return net


class TFComponentVal:
    pass

class Identity(TFComponent):
    def __init__(self, **kwargs):
        self.noDirect = True
        self.opts = {
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        return incoming


class Sequence(TFComponent):
    def __init__(self, blocks, **kwargs):
        self.noDirect = True
        self.blocks = blocks
        self.opts = {
            'name': "Sequence"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        resnet = incoming
        with tf.name_scope(opts.name):
            for blk in self.blocks:
                resnet = blk[resnet]
        return resnet


class ParallelVal(TFComponentVal):
    def __init__(self, opts, inherit, scope):
        self.blocks = list()
        self.opts = opts
        self.inherit = inherit
        self.scope = scope

    def resolve(self):
        opts = self.opts
        inherit = self.inherit
        with tf.name_scope(self.scope):
            is_training = tflearn.get_training_mode()
            blocks = tf.pack(self.blocks)
            basic = tf.reduce_sum(blocks, 0)
            oneChoice = tf.random_uniform([], maxval=len(self.blocks), dtype='int32')
            one = tf.cond(is_training, lambda: tf.gather(blocks,oneChoice), lambda: basic)
            someChoice = tf.less(tf.random_uniform([len(self.blocks)]), inherit['localDroppathProb'])
            some = tf.cond(is_training, lambda: tf.reduce_sum(tf.boolean_mask(blocks,someChoice), 0), lambda: basic)
            some = tf.cond(tf.reduce_any(someChoice), lambda: some, lambda: one)
            resnet = condition(inherit['globalDroppath'], one, condition(inherit['localDroppath'], some, basic))
        return resnet



class Parallel(TFComponent):
    def __init__(self, blocks, **kwargs):
        self.noDirect = True
        self.blocks = blocks
        self.opts = {
            'name': "Parallel"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        resnet = incoming
        with tf.name_scope(opts.name) as scope:
            blocksMixed = [blk[resnet] for blk in self.blocks]
            blocks = ParallelVal(opts, inherit, scope)
            for blk in blocksMixed:
                if isinstance(blk, ParallelVal):
                    blocks.blocks = blocks.blocks + blk.blocks
                else:
                    blocks.blocks.append(blk)
            return blocks


class Chain(TFComponent):
    def __init__(self, size, block, **kwargs):
        self.noDirect = True
        self.size = size
        self.block = block
        self.opts = {
            'name': "Chain"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        resnet = incoming
        with tf.name_scope(opts.name):
            for i in range(self.size):
                resnet = self.block[resnet]
        return resnet


class Fractal(TFComponent):
    def __init__(self, size, block, **kwargs):
        self.noDirect = True
        self.size = size
        self.block = block
        self.opts = {
            'name': "Fractal"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        resnet = incoming
        with tf.name_scope(opts.name):
            if self.size <= 1:
                return self.block[resnet]
            else:
                sub = Fractal(self.size-1, self.block)
                resnet = Parallel([self.block, Chain(2, sub)])[resnet]
        return resnet


class Residual(TFComponent):
    def __init__(self, block, **kwargs):
        self.noDirect = True
        self.block = block
        self.opts = {
            'name': "Residual"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
        resnet = incoming
        with tf.name_scope(opts.name):
            resnet = Parallel([Identity(), self.block])
        return resnet


class Conv2d(TFComponent):
    def __init__(self, nb_filter, filter_size, **kwargs):
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.opts = {
            'strides': 1,
            'padding': 'same',
            'activation': 'linear',
            'bias': True,
            'weights_init': 'uniform_scaling',
            'bias_init': 'zeros',
            'regularizer': None,
            'weight_decay': 0.001,
            'trainable': True,
            'restore': True,
            'name': "Conv2D"
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):

        assert opts.padding in ['same', 'valid', 'SAME', 'VALID'], \
            "Padding must be same' or 'valid'"

        input_shape = utils.get_incoming_shape(incoming)
        assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
        filter_size = utils.autoformat_filter_conv2d(self.filter_size,
                                                     input_shape[-1],
                                                     self.nb_filter)
        strides = utils.autoformat_kernel_2d(opts.strides)
        padding = utils.autoformat_padding(opts.padding)

        with tf.name_scope(opts.name) as scope:

            W_init = opts.weights_init
            if isinstance(opts.weights_init, str):
                W_init = initializations.get(opts.weights_init)()
            W_regul = None
            if opts.regularizer:
                W_regul = lambda x: losses.get(opts.regularizer)(x, opts.weight_decay)
            W = vs.variable(scope + 'W', shape=filter_size,
                            regularizer=W_regul, initializer=W_init,
                            trainable=opts.trainable, restore=opts.restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

            b = None
            if opts.bias:
                b_init = initializations.get(opts.bias_init)()
                b = vs.variable(scope + 'b', shape=self.nb_filter,
                                initializer=b_init, trainable=opts.trainable,
                                restore=opts.restore)
                # Track per layer variables
                tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)

            inference = tf.nn.conv2d(incoming, W, strides, padding)
            if b: inference = tf.nn.bias_add(inference, b)

            if isinstance(opts.activation, str):
                inference = activations.get(opts.activation)(inference)
            elif hasattr(activation, '__call__'):
                inference = activation(inference)
            else:
                raise ValueError("Invalid Activation.")

            # Track activations.
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

        # Add attributes to Tensor to easy access weights.
        inference.scope = scope
        inference.W = W
        inference.b = b

        return inference


class ShallowResidualBlock(TFComponent):
    def __init__(self, out_channels, **kwargs):
        self.out_channels = out_channels
        self.opts = {
            'downsample': False,
            'downsample_strides': 2,
            'activation': 'relu',
            'batch_norm': True,
            'bias': True,
            'weights_init': 'variance_scaling',
            'bias_init': 'zeros',
            'regularizer': 'L2',
            'weight_decay': 0.0001,
            'trainable': True,
            'restore': True,
            'name': 'ResidualBlock'
        }
        self.opts.update(kwargs)

    def get(self, incoming, opts, inherit):
            resnet = incoming
            in_channels = incoming.get_shape().as_list()[-1]

            with tf.name_scope(opts.name):

                identity = resnet

                if not downsample:
                    opts.downsample_strides = 1

                if opts.batch_norm:
                    resnet = tflearn.batch_normalization(resnet)
                resnet = tflearn.activation(resnet, opts.activation)

                resnet = conv_2d(resnet, self.out_channels, 3,
                                 opts.downsample_strides, 'same', 'linear',
                                 opts.bias, opts.weights_init, opts.bias_init,
                                 opts.regularizer, opts.weight_decay, opts.trainable,
                                 opts.restore)

                if opts.batch_norm:
                    resnet = tflearn.batch_normalization(resnet)
                resnet = tflearn.activation(resnet, opts.activation)

                resnet = conv_2d(resnet, self.out_channels, 3, 1, 'same',
                                 'linear', opts.bias, opts.weights_init,
                                 opts.bias_init, opts.regularizer, opts.weight_decay,
                                 opts.trainable, opts.restore)

                # Downsampling
                if opts.downsample_strides > 1:
                    identity = tflearn.avg_pool_2d(identity, 1,
                                                   opts.downsample_strides)

                # Projection to new dimension
                if in_channels != self.out_channels:
                    ch = (self.out_channels - in_channels)//2
                    identity = tf.pad(identity,
                                      [[0, 0], [0, 0], [0, 0], [ch, ch]])
                    in_channels = self.out_channels

                #resnet = resnet + identity

            return resnet
