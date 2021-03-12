from imutils import paths, build_montages, resize
from PIL import Image
import os
import cv2
import numpy as np

path = './output/'
NbImages = 100
pas = 500

paths = list(paths.list_files(path))

generated = []
plots = []
for p in range(1, NbImages+1):
    name = os.path.join(path, f'generated/Epoch_{p*pas}.jpg')
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    generated.append(img)
    name = os.path.join(path, f'plots/loss_plots_epoch_{p*pas}.png')
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plots.append(img)

frames = []
for i in range(NbImages):
    gen = generated[i]
    h, w = gen.shape[0], gen.shape[1]
    plo = resize(plots[i], height=h)
    gen = np.array(gen)
    plo = np.array(plo)
    frame = np.concatenate((gen, plo), axis = 1).astype('uint8')
    frame = Image.fromarray(frame)
    frames.append(frame)

frames[0].save('animated.gif', save_all = True, append_images = frames[1:], optimize = False, duration = 40, loop=0)