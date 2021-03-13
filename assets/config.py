# initialize useful variables

# learning rates of the discriminator and the global model
INIT_LR_DISC = 1e-4
INIT_LR_GAN = 7e-5
# shape of the generated images (square, multiple of 80)
IMAGE_SHAPE = (160, 160)
# shape of the noise based on which images will be generated
NOISE_RES = 100
# number of images to be generated in the same time
IMG_VISU = 1