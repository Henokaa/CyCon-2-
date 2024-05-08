import base64

import pandas as pd
import numpy as np

from backend.Classes.NeuralNetworkFiles import Layer
from backend.Classes import HyperParameters

import matplotlib.pyplot as plt

from tensorflow import keras

# Create a list of core layers to select from.
list_Convolution = []

def getConvolution():
    return list_Convolution

## Conv1D Layer
Name = "Conv1D"
Display_Name = "1D"
Definition = ['1D convolution layer (e.g. temporal convolution).\nThis layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.\n\nWhen using this layer as the first layer in a model, provide an input_shape argument (tuple of integers or None, e.g. (10, 128) for sequences of 10 vectors of 128-dimensional vectors, or (None, 128) for variable-length sequences of 128-dimensional vectors.Parameter_0 = {"Name":"shape_x", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"]'],
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of a single integer, specifying the length of the 1D convolution window."}
Parameter_2 = {"Name":"strides", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."}
Parameter_3 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_4 = {"Name":"data_format", "Type": ["option"], "Default_option":"channels_last", "Default_value":"channels_last", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_5 = {"Name":"dilation_rate", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1."}
Parameter_6 = {"Name":"groups", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups."}
Parameter_7 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_8 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_9 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_10 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_11 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_12 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_13 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


## Conv2D Layer
Name = "Conv2D"
Display_Name = "2D"
Definition = ['2D convolution layer (e.g. spatial convolution over images).\nThis layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.\n\nWhen using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last". You can use None when a dimension has variable size.']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_2 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_3 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_6 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_7 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_8 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_9 = {"Name":"groups", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups."}
Parameter_10 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_11 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_12 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_13 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_14 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_15 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_16 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_17 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_18 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


## Conv3D Layer
Name = "Conv3D"
Display_Name = "3D"
Definition = ['3D convolution layer (e.g. spatial convolution over volumes).\nThis layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.\n\nWhen using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 128, 1) for 128x128x128 volumes with a single channel, in data_format="channels_last".']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_2 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_3 = {"Name":"kernel_size_Z", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along each spatial dimension. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nZ is the third integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_6 = {"Name":"strides_Z", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nZ is the third integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_7 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_8 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_9 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nFor single integer simply fill in the X section."}
Parameter_10 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_11 = {"Name":"dilation_rate_Z", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nZ is the third integer in a tuple of shape of (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_12 = {"Name":"groups", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups."}
Parameter_13 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_14 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_15 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_16 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_17 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_18 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_19 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_17 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_18 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16, "Parameter_17":Parameter_17, "Parameter_18":Parameter_18, "Parameter_19":Parameter_19}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))

## SeparableConv1D Layer
Name = "SeparableConv1D"
Display_Name = "Separable Convolutional 1D"
Definition = ['Depthwise separable 1D convolution.\nThis layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels. If use_bias is True and a bias initializer is provided, it adds a bias vector to the output. It then optionally applies an activation function to produce the final output.']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"A single integer specifying the spatial dimensions of the filters."}
Parameter_2 = {"Name":"strides", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."}
Parameter_3 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_4 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_5 = {"Name":"dilation_rate", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"A single integer, specifying the dilation rate to use for dilated convolution."}
Parameter_6 = {"Name":"depth_multiplier", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."}
Parameter_7 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_8 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_9 = {"Name":"depthwise_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the depthwise convolution kernel (see keras.initializers). If None, then the default initializer ('glorot_uniform') will be used."}
Parameter_10 = {"Name":"pointwise_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the pointwise convolution kernel (see keras.initializers). If None, then the default initializer ('glorot_uniform') will be used."}
Parameter_11 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the bias vector. If None, the default initializer ('zeros') will be used (see keras.initializers)."}
Parameter_12 = {"Name":"depthwise_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Optional regularizer for the depthwise convolution kernel (see keras.regularizers)."}
Parameter_13 = {"Name":"pointwise_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Optional regularizer for the pointwise convolution kernel (see keras.regularizers)."}
Parameter_14 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_15 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"depthwise_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training (see keras.constraints)."}
#Parameter_12 = {"Name":"pointwise_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Optional projection function to be applied to the pointwise kernel after being updated by an Optimizer (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Optional projection function to be applied to the bias after being updated by an Optimizer (see keras.constraints)."}
Parameter_16 = {"Name":"trainable", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, if True the weights of this layer will be marked as trainable (and listed in layer.trainable_weights)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))

## SeparableConv2D Layer
Name = "SeparableConv2D"
Display_Name = "Separable Convolutional 2D"
Definition = ['Depthwise separable 2D convolution.\nSeparable convolutions consist of first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes the resulting output channels. The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.\n\nIntuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y)."}
Parameter_2 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the second integer in a tuple of shape (X,Y)."}
Parameter_3 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_6 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_7 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_8 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_9 = {"Name":"depth_multiplier", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."}
Parameter_10 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_11 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_12 = {"Name":"depthwise_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the depthwise convolution kernel (see keras.initializers). If None, then the default initializer ('glorot_uniform') will be used."}
Parameter_13 = {"Name":"pointwise_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the pointwise convolution kernel (see keras.initializers). If None, then the default initializer ('glorot_uniform') will be used."}
Parameter_14 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the bias vector. If None, the default initializer ('zeros') will be used (see keras.initializers)."}
Parameter_15 = {"Name":"depthwise_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Optional regularizer for the depthwise convolution kernel (see keras.regularizers)."}
Parameter_16 = {"Name":"pointwise_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Optional regularizer for the pointwise convolution kernel (see keras.regularizers)."}
Parameter_17 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_18 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"depthwise_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training (see keras.constraints)."}
#Parameter_12 = {"Name":"pointwise_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Optional projection function to be applied to the pointwise kernel after being updated by an Optimizer (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Optional projection function to be applied to the bias after being updated by an Optimizer (see keras.constraints)."}
Parameter_19 = {"Name":"trainable", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, if True the weights of this layer will be marked as trainable (and listed in layer.trainable_weights)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16, "Parameter_17":Parameter_17, "Parameter_18":Parameter_18, "Parameter_19":Parameter_19}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))

## DepthwiseConv2D Layer
Name = "DepthwiseConv2D"
Display_Name = "Depthwise Convolutional 2D"
Definition = ['Depthwise 2D convolution.\nDepthwise convolution is a type of convolution in which each input channel is convolved with a different kernel (called a depthwise kernel). You can understand depthwise convolution as the first step in a depthwise separable convolution.\n\nIt is implemented via the following steps:\n\nSplit the input into individual channels.\nConvolve each channel with an individual depthwise kernel with depth_multiplier output channels.\nConcatenate the convolved outputs along the channels axis.\n\nUnlike a regular 2D convolution, depthwise convolution does not mix information across different input channels.\n\nThe depth_multiplier argument determines how many filter are applied to one input channel. As such, it controls the amount of output channels that are generated per input channel in the depthwise step.']
Parameter_0 = {"Name":"filters_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_1 = {"Name":"filters_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_2 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_3 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_6 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_7 = {"Name":"depth_multiplier", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier."}
Parameter_8 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_9 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_10 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_11 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_12 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_13 = {"Name":"depthwise_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the depthwise convolution kernel (see keras.initializers). If None, then the default initializer ('glorot_uniform') will be used."}
Parameter_14 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"An initializer for the bias vector. If None, the default initializer ('zeros') will be used (see keras.initializers)."}
Parameter_15 = {"Name":"depthwise_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Optional regularizer for the depthwise convolution kernel (see keras.regularizers)."}
Parameter_16 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_17 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"depthwise_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Optional projection function to be applied to the depthwise kernel after being updated by an Optimizer (e.g. used for norm constraints or value constraints for layer weights). The function must take as input the unprojected variable and must return the projected variable (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Optional projection function to be applied to the bias after being updated by an Optimizer (see keras.constraints)."}
Parameter_18 = {"Name":"trainable", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, if True the weights of this layer will be marked as trainable (and listed in layer.trainable_weights)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16, "Parameter_17":Parameter_17, "Parameter_18":Parameter_18}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


## Conv1DTranspose Layer
Name = "Conv1DTranspose"
Display_Name = "Convolution 1D Transpose"
Definition = ['Transposed convolution layer (sometimes called Deconvolution).\nThe need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.\n\nWhen using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 3) for data with 128 time steps and 3 channels.']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of a single integer, specifying the length of the 1D convolution window."}
Parameter_2 = {"Name":"strides", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1."}
Parameter_3 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_4 = {"Name":"output_padding", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer specifying the amount of padding along the time dimension of the output tensor. The amount of output padding must be lower than the stride. If set to None (default), the output shape is inferred.'}
Parameter_5 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_6 = {"Name":"dilation_rate", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1."}
Parameter_7 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_8 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_9 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_10 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_11 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_12 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_13 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))

## Conv2DTranspose Layer
Name = "Conv2DTranspose"
Display_Name = "Convolution 2D Transpose"
Definition = ['Transposed convolution layer (sometimes called Deconvolution).\nThe need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.\n\nWhen using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y)."}
Parameter_2 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the second integer in a tuple of shape (X,Y)."}
Parameter_3 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_6 = {"Name":"output_padding_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer or tuple/list of 2 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.\n\nX is the first integer in a tuple of shape (X,Y).'}
Parameter_7 = {"Name":"output_padding_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer or tuple/list of 2 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.\n\nY is the first integer in a tuple of shape (X,Y).'}
Parameter_8 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_9 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_10 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y).\n\nFor single integer simply fill in the X section."}
Parameter_11 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_12 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_13 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_14 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_15 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_16 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_17 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16, "Parameter_17":Parameter_17}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))

## Conv3DTranspose Layer
Name = "Conv3DTranspose"
Display_Name = "Convolution 3D Transpose"
Definition = ['Transposed convolution layer (sometimes called Deconvolution).\nThe need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution.\n\nWhen using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 128, 3) for a 128x128x128 volume with 3 channels if data_format="channels_last".']
Parameter_0 = {"Name":"filters", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)."}
Parameter_1 = {"Name":"kernel_size_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nX is the first integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_2 = {"Name":"kernel_size_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nY is the second integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_3 = {"Name":"kernel_size_Z", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.\n\nZ is the third integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_4 = {"Name":"strides_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nX is the first integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_5 = {"Name":"strides_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the second integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_6 = {"Name":"strides_Z", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"An integer or tuple/list of 3 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.\n\nY is the third integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_7 = {"Name":"padding", "Type": ["option"], "Default_option":"valid", "Default_value":"valid", "Possible":["valid","same","causal"],
             "Definition":'One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.'}
Parameter_8 = {"Name":"output_padding_X", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer or tuple/list of 3 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.\n\nX is the first integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section.'}
Parameter_9 = {"Name":"output_padding_Y", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer or tuple/list of 3 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.\n\nY is the second integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section.'}
Parameter_10 = {"Name":"output_padding_Z", "Type": ["int"], "Default_option":"", "Default_value":"", "Possible":["int"],
             "Definition":'An integer or tuple/list of 3 integers, specifying the amount of padding along the height and width of the output tensor. Can be a single integer to specify the same value for all spatial dimensions. The amount of output padding along a given dimension must be lower than the stride along that same dimension. If set to None (default), the output shape is inferred.\n\nZ is the third integer in a tuple of shape (X,Y,Z).\n\nFor single integer simply fill in the X section.'}
Parameter_11 = {"Name":"data_format", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":["channels_last", "channels_first", "None"],
             "Definition":"A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, width). Note that the channels_first format is currently not supported by TensorFlow on CPU."}
Parameter_12 = {"Name":"dilation_rate_X", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nX is the first integer in a tuple of shape of (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_13 = {"Name":"dilation_rate_Y", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nY is the second integer in a tuple of shape of (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_14 = {"Name":"dilation_rate_Z", "Type": ["int"], "Default_option":1, "Default_value":1, "Possible":["int"],
             "Definition":"an integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.\n\nZ is the third integer in a tuple of shape of (X,Y,Z).\n\nFor single integer simply fill in the X section."}
Parameter_15 = {"Name":"activation", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getActivations(),
             "Definition":"Activation function to use. If you don't specify anything, no activation is applied (see keras.activations)."}
Parameter_16 = {"Name":"use_bias", "Type": ["bool"], "Default_option":True, "Default_value":True, "Possible":[True,False],
             "Definition":"Boolean, whether the layer uses a bias vector."}
Parameter_17 = {"Name":"kernel_initializer", "Type": ["option"], "Default_option":"glorot_uniform", "Default_value":"glorot_uniform", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_18 = {"Name":"bias_initializer", "Type": ["option"], "Default_option":"zeros", "Default_value":"zeros", "Possible":Layer.getInitializers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_19 = {"Name":"kernel_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the kernel weights matrix (see keras.regularizers)."}
Parameter_20 = {"Name":"bias_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the bias vector (see keras.regularizers)."}
Parameter_21 = {"Name":"activity_regularizer", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getRegularizers(),
             "Definition":"Regularizer function applied to the output of the layer (its 'activation') (see keras.regularizers)."}
#Parameter_12 = {"Name":"kernel_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints,
#             "Definition":"Constraint function applied to the kernel matrix (see keras.constraints)."}
#Parameter_13 = {"Name":"bias_constraint", "Type": ["option"], "Default_option":"None", "Default_value":"None", "Possible":Layer.getConstraints(),
#             "Definition":"Constraint function applied to the bias vector (see keras.constraints)."}


Parameters = {"Parameter_0":Parameter_0, "Parameter_1":Parameter_1, "Parameter_2":Parameter_2, "Parameter_3":Parameter_3,
              "Parameter_4":Parameter_4, "Parameter_5":Parameter_5, "Parameter_6":Parameter_6, "Parameter_7":Parameter_7,
              "Parameter_8":Parameter_8, "Parameter_9":Parameter_9, "Parameter_10":Parameter_10, "Parameter_11":Parameter_11,
              "Parameter_12":Parameter_12, "Parameter_13":Parameter_13, "Parameter_14":Parameter_14, "Parameter_15":Parameter_15,
              "Parameter_16":Parameter_16, "Parameter_17":Parameter_17, "Parameter_18":Parameter_18, "Parameter_19":Parameter_19,
              "Parameter_20":Parameter_20, "Parameter_21":Parameter_21}

list_Convolution.append(Layer.Layer(Name, Display_Name, Definition, Parameters))


def create_Layer(data, i):
    # get layer name
    layer = data["Layer_" + str(i)]

    # get the chosen settings for the layer
    Parameters = HyperParameters.getParameters(data["Layer_" + str(i)], list_Convolution)
    settings = HyperParameters.getSettings(data, Parameters, i, Layer.getName())

    new_layer = ""

    ## Conv1D Layer
    if layer == "Conv1D":

        # Create the layer
        new_layer = keras.layers.Conv1D(filters=settings["Parameter_0"], kernel_size=settings["Parameter_1"], strides=settings["Parameter_2"],
                                        padding=settings["Parameter_3"], data_format=settings["Parameter_4"], dilation_rate=settings["Parameter_5"],
                                        groups=settings["Parameter_6"], activation=settings["Parameter_7"], use_bias=settings["Parameter_8"],
                                        kernel_initializer=settings["Parameter_9"], bias_initializer=settings["Parameter_10"], kernel_regularizer=settings["Parameter_11"],
                                        bias_regularizer=settings["Parameter_12"], activity_regularizer=settings["Parameter_13"])
    
    ## Conv2D Layer
    if layer == "Conv2D":
        # Kernel_size
        if settings["Parameter_2"] != None and settings["Parameter_2"] != "":
            kernel_size = (settings["Parameter_1"],settings["Parameter_2"])
        else:
            kernel_size = settings["Parameter_1"]
        
        # strides
        if settings["Parameter_4"] != None and settings["Parameter_4"] != "":
            strides = (settings["Parameter_3"],settings["Parameter_4"])
        else:
            strides = settings["Parameter_3"]

        # dilation_rate
        if settings["Parameter_8"] != None and settings["Parameter_8"] != "":
            dilation_rate = (settings["Parameter_7"],settings["Parameter_8"])
        else:
            dilation_rate = settings["Parameter_7"]


        # Create the layer
        new_layer = keras.layers.Conv2D(filters=settings["Parameter_0"], kernel_size=kernel_size, strides=strides,
                                        padding=settings["Parameter_5"], data_format=settings["Parameter_6"], dilation_rate=dilation_rate,
                                        groups=settings["Parameter_9"], activation=settings["Parameter_10"], use_bias=settings["Parameter_11"],
                                        kernel_initializer=settings["Parameter_12"], bias_initializer=settings["Parameter_13"], kernel_regularizer=settings["Parameter_14"],
                                        bias_regularizer=settings["Parameter_15"], activity_regularizer=settings["Parameter_16"])



    ## Conv3D Layer
    if layer == "Conv3D":

        # Kernel_size
        if settings["Parameter_2"] != None and settings["Parameter_2"] != "" and settings["Parameter_3"] != None and settings["Parameter_3"] != "":
            kernel_size = (settings["Parameter_1"],settings["Parameter_2"], settings["Parameter_3"])
        else:
            kernel_size = settings["Parameter_1"]
        
        # strides
        if settings["Parameter_5"] != None and settings["Parameter_5"] != "" and settings["Parameter_6"] != None and settings["Parameter_6"] != "":
            strides = (settings["Parameter_4"],settings["Parameter_5"],settings["Parameter_6"])
        else:
            strides = settings["Parameter_4"]

        # dilation_rate
        if settings["Parameter_10"] != None and settings["Parameter_10"] != "" and settings["Parameter_11"] != None and settings["Parameter_11"] != "":
            dilation_rate = (settings["Parameter_9"],settings["Parameter_10"],settings["Parameter_11"])
        else:
            dilation_rate = settings["Parameter_9"]


        # Create the layer
        new_layer = keras.layers.Conv3D(filters=settings["Parameter_0"], kernel_size=kernel_size, strides=strides,
                                        padding=settings["Parameter_7"], data_format=settings["Parameter_8"], dilation_rate=dilation_rate,
                                        groups=settings["Parameter_12"], activation=settings["Parameter_13"], use_bias=settings["Parameter_14"],
                                        kernel_initializer=settings["Parameter_15"], bias_initializer=settings["Parameter_16"], kernel_regularizer=settings["Parameter_17"],
                                        bias_regularizer=settings["Parameter_18"], activity_regularizer=settings["Parameter_19"])



    ## SeparableConv1D Layer


    ## SeparableConv2D Layer


    ## Depthwise2D Layer


    ## Conv1DTranspose Layer


    ## Conv2DTranspose Layer


    ## Conv3DTranspose Layer
    

        
    return new_layer