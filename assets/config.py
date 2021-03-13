# initialize useful variables

# learning rates of the discriminator and the global model
INIT_LR_DISC = 1e-4
INIT_LR_GAN = 1e-4
# shape of the generated images (square, multiple of 80)
IMAGE_SHAPE = (320, 320)
# shape of the noise based on which images will be generated
NOISE_RES = 200
# number of images to be generated in the same time
IMG_VISU = 1