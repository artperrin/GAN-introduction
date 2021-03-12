# import the necessary packages
from assets.tools import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from assets.dcgan import DCGAN
import logging as lg
from assets import config
import argparse

lg.getLogger().setLevel(lg.INFO)

gpu_computation_activate()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, 
    help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=1000,
    help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=32,
    help="batch size for training")
ap.add_argument("-g", "--grayscale", action='store_true', 
    help="to use grayscale database")
ap.add_argument("-f", "--face-detection", action='store_true',
    help="if the user wants to perform face detection")
ap.add_argument("-l", "--limit-images", type=int,
    help="maximum images of the dataset to be considered")
ap.add_argument("-o", "--output-generated", action='store_true',
    help="output intermediate generated images")
ap.add_argument("-p", "--output-plots", action='store_true',
    help="output intermediate training plots")
ap.add_argument("-m", "--model-output", type=str, default=None,
    help="path to output of the models")
args = vars(ap.parse_args())

# store useful variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
GRAYSCALE = args["grayscale"]
if args["output_generated"]:
    generated_path = './output/generated'
if args["output_plots"]:
    plots_path = './output/plots'

INIT_LR_DISC = config.INIT_LR_DISC
INIT_LR_GAN = config.INIT_LR_GAN
W, H = config.image_shape[0], config.image_shape[1]

global_time = timer()
# load the dataset
dataset = dataset_handler(args["dataset"])

dataset.load(args["grayscale"], args["face_detection"], args["limit_images"])

trainImages = dataset.pre_process()

if len(trainImages) < BATCH_SIZE:
    lg.warning('Not enough data for the batch, change batch size or expand dataset!')

if GRAYSCALE:
    NbChannels = 1
else:
    NbChannels = 3

# build the generator
lg.info("Building generator and discriminator...")
gen = DCGAN.build_generator(W, 64, channels=NbChannels)
# build the discriminator
disc = DCGAN.build_discriminator(W, H, NbChannels)
discOpt = Adam(lr=INIT_LR_DISC, beta_1=0.5, decay=INIT_LR_DISC / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

# build the adversarial model by first setting the discriminator to
# *not* be trainable, then combine the generator and discriminator
# together
lg.info("Building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)
# compile the GAN
ganOpt = Adam(lr=INIT_LR_GAN, beta_1=0.5, decay=INIT_LR_GAN / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=ganOpt)

# randomly generate some benchmark noise so we can consistently
# visualize how the generative modeling is learning
lg.info("Starting training...")

# setting useful variable
batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

model = dcgan(gen, disc, gan, BATCH_SIZE, NUM_EPOCHS, batchesPerEpoch)

model.train(trainImages, generated_path, plots_path=plots_path)

model.export_training_data(plots_path)

if args["model_output"] is not None:
    model.save_global(args["model_output"])

global_time.update()
lg.info(f"Process ended within {global_time.get_time()}.")