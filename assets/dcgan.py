# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU') # to compute with GPU : not working for me

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, inputDim=100, outputDim=512):
      """builds the generator model

      Parameters
      ----------
      dim : int
          wanted dimension of the generated image (square), must be multiple of 80 (3 upsampling layers)
      depth : int
          depth of the wanted volume
      channels : int, optional
          number of channels of the images, by default 1
      inputDim : int, optional
          input dimension of the noise based on which images will be generated, by default 100
      outputDim : int, optional
          output dimension after first FC layer, by default 512

      Returns
      -------
      tensorflow.python.keras.engine.sequential.Sequential
          model of the generator
      """
      dim = dim//8
      # initialize the model along with the input shape to be
      # "channels last" and the channels dimension itself
      model = Sequential()
      inputShape = (dim, dim, depth)
      chanDim = -1
      # first set of FC => RELU => BN layers
      model.add(Dense(input_dim=inputDim, units=outputDim))
      model.add(Activation("relu"))
      model.add(BatchNormalization())
      # second set of FC => RELU => BN layers, this time preparing
      # the number of FC nodes to be reshaped into a volume
      model.add(Dense(dim * dim * depth))
      model.add(Activation("relu"))
      model.add(BatchNormalization())
      # reshape the output of the previous layer set, upsample +
      # apply a transposed convolution, RELU, and BN
      model.add(Reshape(inputShape))
      model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
      model.add(Activation("relu"))
      model.add(BatchNormalization(axis=chanDim))
      # apply other upsample and transposed convolution
      model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),	padding="same"))
      model.add(Activation("relu"))
      model.add(BatchNormalization(axis=chanDim))
      model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),	padding="same"))
      model.add(Activation("relu"))
      model.add(BatchNormalization(axis=chanDim))
      # this time output the TANH activation
      model.add(Activation("tanh"))
      # return the generator model
      return model

    @staticmethod
    def build_discriminator(width, height, depth=1, alpha=0.2):
      """builds the generator model

      Parameters
      ----------
      width : int
          width of the input image
      height : int
          height of the input image
      depth : int, optional
          number of channels of the input image, by default 1
      alpha : float, optional
          [description], by default 0.2

      Returns
      -------
      tensorflow.python.keras.engine.sequential.Sequential
          model of the discriminator
      """
      # initialize the model along with the input shape to be
      # "channels last"
      model = Sequential()
      inputShape = (height, width, depth)
      # first set of CONV => RELU layers
      model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2), input_shape=inputShape))
      model.add(LeakyReLU(alpha=alpha))
      # second set of CONV => RELU layers
      model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
      model.add(LeakyReLU(alpha=alpha))
      # first (and only) set of FC => RELU layers
      model.add(Flatten())
      model.add(Dense(512))
      model.add(LeakyReLU(alpha=alpha))
      # sigmoid layer outputting a single value
      model.add(Dense(1))
      model.add(Activation("sigmoid"))
      # return the discriminator model
      return model