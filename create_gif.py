from imutils import paths, resize
from PIL import Image
from os.path import join
import logging as lg
import cv2
import numpy as np
import argparse

lg.getLogger().setLevel(lg.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--output-generated", type=str, required=True,
    help="path to generated images")
ap.add_argument("-o", "--path", type=str, default='./',
    help="path to output animation")
ap.add_argument("-p", "--output-plots", type=str, default=None, 
    help="path to plots if wanted")
args = vars(ap.parse_args())

lg.info('Reading generated files...')
paths_generated = list(paths.list_files(args["output_generated"]))
paths_generated = sorted(paths_generated, key=lambda x:len(x))

generated = []
for path in paths_generated:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    generated.append(img)

frames = [Image.fromarray(np.array(gen)) for gen in generated]

if args["output_plots"] is not None:
    lg.info('Reading plots files...')
    h, w = generated[0].shape[0], generated[0].shape[1]
    paths_plots = list(paths.list_files(args["output_plots"]))
    paths_plots = sorted(paths_plots, key=lambda x:len(x))
    plots = []

    for path in paths_plots:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plots.append(img)
    frames = []

    lg.info('Formatting output images...')
    for i in range(len(generated)):
        gen = generated[i]
        gen = np.array(gen)
        plo = resize(plots[i], height=h)
        plo = np.array(plo)
        frame = np.concatenate((gen, plo), axis = 1).astype('uint8')
        frame = Image.fromarray(frame)
        frames.append(frame)

lg.info('Exporting animation...')
frames[0].save(join(args["path"], 'animation.gif'), save_all = True, append_images = frames[1:], optimize = False, duration = 40, loop=0)