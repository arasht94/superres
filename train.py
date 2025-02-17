import random
import glob
import subprocess
import sys
import os

from tensorflow.train import AdamOptimizer
from PIL import Image
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, TensorBoard
from tensorflow.python.keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from model import DefaulModel, DCSCNModel, WDSRModelA, WDSRModelB
from datetime import datetime


run = wandb.init(project='superres')
config = run.config

config.num_epochs = 100
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.input_depth = 3
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)


###########################################################################
args = sys.argv

if len(args) != 2:
    print("Error: need to specify model.")
    print("Usage: python train.py <MODEL>")

model_type = args[1]
if model_type == 'default':
    model_class = DefaulModel
elif model_type == 'dcscn':
    model_class = DCSCNModel
elif model_type == 'wdsra':
    model_class = WDSRModelA
elif model_type == 'wdsrb':
    model_class = WDSRModelB
else:
    raise ValueError("Error: unrecognized model: {}".format(model_type))

input_shape = (config.input_height, config.input_width, config.input_depth)
model = model_class(input_shape=input_shape)


# Tensorboard
# timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# tensorboard_dir = os.path.join('./tensorboard', timestamp)
# os.makedirs(tensorboard_dir)
# tensorboard = TensorBoard(log_dir=tensorboard_dir)
optimizer = Adam(1e-3)
###########################################################################
# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer=optimizer, loss='mae',
              metrics=[perceptual_distance])
model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)