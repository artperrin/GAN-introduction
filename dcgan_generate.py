# import the necessary packages
from dcgan import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
from imutils import paths
from autocrop import Cropper
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

W, H = 160, 160
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default='./output',
    help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=1000,
    help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=32,
    help="batch size for training")
args = vars(ap.parse_args())

# store the epochs and batch size in convenience variables, then
# initialize our learning rate
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4

# load the Fashion MNIST dataset and stack the training and testing
# data points so we have additional training data

print("[INFO] loading dataset...")

# ((trainX, _), (testX, _)) = mnist.load_data()
# trainImages = np.concatenate([trainX, testX])

imagePaths = paths.list_files('./data/')
cropper = Cropper(width=W, height=H, face_percent=70)

trainImages = []
for path in imagePaths:
    img = cropper.crop(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        continue

    # cv2.imshow('out',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    trainImages.append(img)
trainImages = np.array(trainImages)

# add in an extra dimension for the channel and scale the images
# into the range [-1, 1] (which is the range of the tanh
# function)
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype('float') - 127.5) / 127.5

# build the generator
print("[INFO] building generator...")
gen = DCGAN.build_generator(10, 64, channels=1)
# build the discriminator
print("[INFO] building discriminator...")
disc = DCGAN.build_discriminator(W, H, 1)
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

# build the adversarial model by first setting the discriminator to
# *not* be trainable, then combine the generator and discriminator
# together
print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)
# compile the GAN
ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)

# randomly generate some benchmark noise so we can consistently
# visualize how the generative modeling is learning
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(4, 100))

# setting useful variables
batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)
fakeLabels = [1] * BATCH_SIZE
fakeLabels = np.reshape(fakeLabels, (-1,))

# loop over the epochs
for epoch in range(0, NUM_EPOCHS):
    # loop over the batches
    for i in range(0, batchesPerEpoch):
        # initialize an (empty) output path
        p = None
        # select the next batch of images, then randomly generate
        # noise for the generator to predict on
        imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        # generate images using the noise + generator model
        genImages = gen.predict(noise, verbose=0)
        # concatenate the *actual* images and the *generated* images,
        # construct class labels for the discriminator, and shuffle
        # the data
        X = np.concatenate((imageBatch, genImages))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        y = np.reshape(y, (-1,))
        (X, y) = shuffle(X, y)
        # train the discriminator on the data
        discLoss = disc.train_on_batch(X, y)
        # let's now train our generator via the adversarial model by
        # (1) generating random noise and (2) training the generator
        # with the discriminator weights frozen
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        ganLoss = gan.train_on_batch(noise, fakeLabels)

    if (epoch+1) % 200 == 0:
        images = gen.predict(benchmarkNoise)
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (W, H), (2, 2))[0]
        cv2.imwrite(os.path.join(args["output"], f'output_epoch_{epoch+1}.jpg'), vis)

    print(f'Epoch {epoch+1} out of {NUM_EPOCHS}', end='\r', flush=True)

print('')

images = gen.predict(benchmarkNoise)
images = ((images * 127.5) + 127.5).astype("uint8")
images = np.repeat(images, 3, axis=-1)
vis = build_montages(images, (W, H), (2, 2))[0]
cv2.imshow('out', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('THE END')