from sklearn.utils import shuffle
from imutils import paths, build_montages
from autocrop import Cropper
from assets import config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging as lg
import time
import cv2
import os

def gpu_computation_activate():
    """To compute with GPU
    """
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

def visualization(images, size, path):
    """to post-process and save generated images as a montage

    Parameters
    ----------
    images : np.array
        non post-processed generated images
    size : tuple
        size of the images
    path : string
        path to save the montage
    """
    images = ((images * 127.5) + 127.5).astype("uint8")
    if images.shape[3]==1: # if the images are grayscale
        images = np.repeat(images, 3, axis=-1)
    vis = build_montages(images, size, (2, 2))[0]
    cv2.imwrite(path, vis)

class timer:
    """Simple chronometer
    """
    def __init__(self):
        self.start = time.time()
        self.current = 0

    def update(self):
        """Updates the time

        Returns
        -------
        float
            time measurement when updated
        """
        self.current = time.time() - self.start
        return self.current
    
    def get_time(self, unit=None, output_as_string=True):
        """Outputs the time at a given unit

        Parameters
        ----------
        unit : string, optional
            's' or 'm' or 'h', the wanted time unit, by default None
        output_as_string : bool, optional
            if the user do not want to output the time as a string, by default True

        Returns
        -------
        string (or float if output_as_string is set to False)
            time measurement
        """
        if unit=='s':
            res = self.current
        elif unit=='m':
            res = self.current/60
        elif unit=='h':
            res = self.current/3600
        else: # automatic display
            res = self.current
            unit = 's'
            if res > 60:
                res /= 60
                unit = 'm'
                if res > 60:
                    res /= 60
                    unit = 'h'
        
        if output_as_string:
            return f'{str(round(res,2)).zfill(5)} {unit}'
        else:
            return round(res,2)

class dataset_handler:
    """Simply the dataset processings
    """
    def __init__(self, path):
        """initialization of the dataset object

        Parameters
        ----------
        path : string
            path to the stored dataset
        """
        self.path = path
        self.image_paths = list(paths.list_files(path))
        self.image_paths = shuffle(self.image_paths)

    def load(self, grayscale=False, face_detection=False, limit=None):
        """loads all the images (grayscale) into the project

        Parameters
        ----------
        grayscale : bool, optional
            convert all images to grayscale if wanted, by default False
        face_detection : bool, optional
            performs face detection and cropping, by default False
        limit : int, optional
            number of maximum images to be loaded (for large datasets), by default None

        Returns
        -------
        np.array
            array of all the loaded images
        """
        self.gray = grayscale
        lg.info("Loading dataset...")
        if face_detection:
            lg.info("...and performing face detection...")
            # the face-cropped image has the wanted shape
            cropper = Cropper(width=config.image_shape[0], height=config.image_shape[1], face_percent=70)
        if limit is not None:
            end = limit
        else:
            end = len(self.image_paths)
        # begin loading
        self.images = []
        for p in range(end):
            path = self.image_paths[p]
            # perform face cropping if wanted
            if face_detection:
                img = cropper.crop(path)
                # if no faces have been detected, skip to the next image
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except cv2.error:
                    continue
            else: # no face cropping, just grayscaling
                img = cv2.imread(path)
                img = cv2.resize(img, (config.image_shape[0], config.image_shape[1]))
            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.images.append(img)
            # draw a progress line
            progress = ''
            for i in range(30):
                if i <= int(p/end*30):
                    progress += '='
                else:
                    progress += '.'
            print(f'{str(p+1).zfill(len(str(end)))}/{end} |{progress}|', end='\r', flush=True)
        print('')
        lg.info(f'Found {len(self.images)} faces, {end - len(self.images)} images removed.')

        return self.images

    def pre_process(self):
        """pre-process the images for them to fit in the training process

        Returns
        -------
        np.array
            array of pre-processed images
        """
        # make sure the type is correct
        self.images = np.array(self.images)
        # add in an extra dimension for the channel and scale the images
        # into the range [-1, 1] (which is the range of the tanh
        # function)
        if self.gray:
            self.images = np.expand_dims(self.images, axis=-1)
        self.images = (self.images.astype('float') - 127.5) / 127.5
        return self.images

class dcgan:
    """Simplify the training process
    """
    def __init__(self, gen, disc, gan, BATCH_SIZE, NUM_EPOCHS, batchesPerEpoch):
        """initialisation of useful variables

        Parameters
        ----------
        gen : tensorflow.python.keras.engine.sequential.Sequential
            model of the generator
        disc : tensorflow.python.keras.engine.sequential.Sequential
            model of the discriminator
        gan : tensorflow.python.keras.engine.functional.Functional
            global model
        BATCH_SIZE : int
            batch size for the training process
        NUM_EPOCHS : int
            number of epochs to train the model
        """
        self.chrono = timer()
        self.gen = gen
        self.disc = disc
        self.gan = gan
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.fakeLabels = [1] * self.BATCH_SIZE
        self.fakeLabels = np.reshape(self.fakeLabels, (-1,))
        self.dbLoss, self.gbLoss = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
        self.DLOSS, self.GLOSS = np.zeros(NUM_EPOCHS), np.zeros(NUM_EPOCHS)
        self.benchmarkNoise = np.random.uniform(-1, 1, size=(4, 100))

    def train(self, trainImages, output_path, NbImages=100, plots_path=None):
        """Trains the discriminator and the global model

        Parameters
        ----------
        trainImages : np.array
            array of images to train the model with
        output_path : string
            path to export generated images
        NbImages : int, optional
            number of images to be exported during training, by default 100
        """
        batchesPerEpoch = int(trainImages.shape[0] / self.BATCH_SIZE)
        W, H = trainImages[0].shape[0], trainImages[0].shape[1]
        end = False
        lg.info('Starting training...')
        # for the user to interrupt the training with ctrl+C and still get results
        try:
            while not end:
                self.DLOSS, self.GLOSS = [], []
                # loop over the epochs
                for epoch in range(0, self.NUM_EPOCHS):
                    eps = timer()
                    # loop over the batches
                    self.dbLoss, self.gbLoss = [], []
                    for i in range(0, batchesPerEpoch):
                        # initialize an (empty) output path
                        p = None
                        # select the next batch of images, then randomly generate
                        # noise for the generator to predict on
                        imageBatch = trainImages[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
                        noise = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, 100))
                        # generate images using the noise + generator model
                        genImages = self.gen.predict(noise, verbose=0)
                        # concatenate the *actual* images and the *generated* images,
                        # construct class labels for the discriminator, and shuffle
                        # the data
                        X = np.concatenate((imageBatch, genImages))
                        y = ([1] * self.BATCH_SIZE) + ([0] * self.BATCH_SIZE)
                        y = np.reshape(y, (-1,))
                        (X, y) = shuffle(X, y)
                        # train the discriminator on the data
                        discLoss = self.disc.train_on_batch(X, y)
                        # train generator via the adversarial model by
                        # (1) generating random noise and (2) training the generator
                        # with the discriminator weights frozen
                        noise = np.random.uniform(-1, 1, (self.BATCH_SIZE, 100))
                        ganLoss = self.gan.train_on_batch(noise, self.fakeLabels)
                        # store loss per batches values
                        self.dbLoss.append(discLoss)
                        self.gbLoss.append(ganLoss)

                    # compute mean losses per batches 
                    meanDL, meanGL = sum(self.dbLoss)/len(self.dbLoss), sum(self.gbLoss)/len(self.gbLoss)
                    # draw a progress line
                    progress = ''
                    for i in range(30):
                        if i <= int((epoch+1)/self.NUM_EPOCHS*30):
                            progress += '='
                        else:
                            progress += '.'

                    # display training statuss
                    self.chrono.update()
                    time_spent = self.chrono.get_time()
                    print(f'Time spent : {time_spent} | {str(round(eps.update(), 2)).zfill(4)} s/epoch | Epoch {str(epoch+1).zfill(len(str(self.NUM_EPOCHS)))}/{self.NUM_EPOCHS} ({str(round((epoch+1)/self.NUM_EPOCHS*100, 2)).zfill(5)} %) |{progress}| mean disc loss = {str(round(meanDL, 4)).zfill(6)} ; mean gan Loss = {str(round(meanGL,4)).zfill(6)}', end='\r', flush=True)
                    # store loss per epochs values
                    self.DLOSS.append(meanDL)
                    self.GLOSS.append(meanGL)
                    
                    # export visualization
                    if self.NUM_EPOCHS < NbImages:
                        images = self.gen.predict(self.benchmarkNoise)
                        visualization(images, (W, H), os.path.join(output_path, f'Epoch_{epoch+1}.jpg'))
                        # save losses as the training goes
                        if plots_path is not None:
                            X = np.zeros(self.NUM_EPOCHS)
                            X[:epoch+1] = self.DLOSS[:]
                            Y = np.zeros(self.NUM_EPOCHS)
                            Y[:epoch+1] = self.GLOSS[:]
                            save_graph(plots_path, epoch, X, Y)
                        
                    else:
                        if (epoch+1) % (self.NUM_EPOCHS//NbImages) == 0:
                            images = self.gen.predict(self.benchmarkNoise)
                            visualization(images, (W, H), os.path.join(output_path, f'Epoch_{epoch+1}.jpg'))
                            # save losses as the training goes
                            if plots_path is not None:
                                X = np.zeros(self.NUM_EPOCHS)
                                X[:epoch+1] = self.DLOSS[:]
                                Y = np.zeros(self.NUM_EPOCHS)
                                Y[:epoch+1] = self.GLOSS[:]
                                save_graph(plots_path, epoch, X, Y)

                # stop the loop
                end = True
            print('')
        # if the training is interrupted
        except KeyboardInterrupt:
            print('')
            lg.info(f'Training interrupted at epoch {epoch+1} ({round((epoch+1)/self.NUM_EPOCHS*100, 2)} %)...')
            # make a last prediction
            images = self.gen.predict(self.benchmarkNoise)
            visualization(images, (W, H), os.path.join(output_path, f'Epoch_{epoch+1}.jpg'))
            pass

    def export_training_data(self, path):
        """save training data as a graph

        Parameters
        ----------
        path : string
            path to save the graph
        """
        lg.info('Exporting training data...')
        out = os.path.join(path, 'loss_plots_total.jpg')
        plt.figure(figsize=(20, 7))
        plt.grid()
        plt.plot(self.DLOSS, color='r', label='disc')
        plt.plot(self.GLOSS, color='b', label='gan')
        plt.legend()
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.title(f'Mean loss (by batches of size {self.BATCH_SIZE}) of both models per epoch')
        plt.savefig(out)
    
    def save_global(self, path):
        """save the model in the h5 format at a given path

        Parameters
        ----------
        path : string
            path to save the h5 models
        """
        lg.info('Saving models...')
        self.gen.save(os.path.join(path, 'generator.h5'))
        self.disc.save(os.path.join(path, 'discriminator.h5'))
        self.gan.save(os.path.join(path, 'GAN.h5'))

def save_graph(plots_path, epoch, X, Y):
    """Save training at intermediate state as a graph

    Parameters
    ----------
    plots_path : string
        path to save the graph
    epoch : int
        epoch for which the graph is plotted
    X : array
        first dataset to be plotted
    Y : array
        second dataset to be plotted
    """
    out = os.path.join(plots_path, f'loss_plots_epoch_{epoch+1}.png')
    plt.figure(figsize=(20, 7))
    plt.grid()
    plt.ylim([0, 20])
    plt.plot(X, color='r', label='disc')
    plt.plot(Y, color='b', label='gan')
    plt.legend()
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title(f'Mean loss (by batches of size of both models per epoch')
    plt.savefig(out)
    plt.close()
