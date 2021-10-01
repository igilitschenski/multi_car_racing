import numpy as np
import tensorflow as tf
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio

"""
Noise generator that helps to explore action in DDPG.
Values are chosen by Ornstein–Uhlenbeck algorithm.
"""

def compute_cholesky_if_possible(x, type_jitter):
    # x is the matrix that we would like to add the jitter
    #and compute cholesky decomposition size(.,.,n,n)
    # type_jitter: 'add' or 'not_add'

    # another way to add jitter:
        # e,v = tf.linalg.eigh(x)
        # eps = 1e-5
        # e = tf.maximum(e, eps)
        # x_modified = tf.matmul(tf.matmul(v,tf.diag(e)), tf.transpose(v))
        # x_modified_chol = tf.cholesky(x_modified)

    if type_jitter == 'add':
        jitter = 1e-2
        cholesky = tf.linalg.cholesky(x + tf.linalg.diag( tf.constant(jitter, dtype=tf.float32, shape=[x.shape[-1]] ) ) )
        # cholesky = torch.linalg.cholesky(x + tf.linalg.diag( tf.constant(jitter, dtype=tf.float32, shape=[x.shape[-1]] ) ) )
    elif type_jitter == 'not_add':
        cholesky = tf.linalg.cholesky(x)
        # cholesky = torch.linalg.cholesky(x)
    return cholesky


def decayed_learning_rate(init_lr_rate, step, decay_steps, decay_rate, staircase=True):
    # returns the exponential decayed learning rate
    # to apply: self.optimizer.learning_rate.assign(self.updated_lr)
    if staircase:
        return init_lr_rate * decay_rate ** int(step / decay_steps)
    else:
        return init_lr_rate * decay_rate ** (step / decay_steps)


def convert_rgb_to_grayscale(image):
    # check the image shape (image is sometimes in the format of (1, x, x, 3))
    if len(image.shape) == 4:
        im = np.reshape(image, (image.shape[1], image.shape[2], image.shape[3]))
    return np.reshape( (0.299 * im[:, :, 0] + 0.587 * im[:, :, 0] + 0.114 * im[:, :, 0]) / 255, (im.shape[0], im.shape[1])) 


def convert_grayscale_to_blackwhite(image):
    black_ind = np.where(image < 0.5)
    white_ind = np.where(image >= 0.5)

    im = image.copy()
    im[black_ind] = 0
    im[white_ind] = 1
    return im


def get_grayscale_from_green(image):
        
    img = image.reshape(-1, 3).astype(np.float32)

    ind_gray = np.where((img[:, 1] < 150) & (img[:, 1] > 100))
    ind_green = np.where(img[:, 1] >= 150)
    ind_red = np.where(img[:, 1] <= 100)

    img[ind_gray[0], 1] = 0.8
    img[ind_green[0], 1] = 0.1
    img[ind_red[0], 1] = 0.5

    return img[:, 1]


def create_gif(frames, interval, dpi, save_path):

    width, height, num_ch = frames[0].shape[-3], frames[0].shape[-2], frames[0].shape[-1]
    with imageio.get_writer(save_path, mode='I') as writer:
        for i in range(len(frames)):
            writer.append_data(frames[i].reshape(width, height, num_ch))

    # fig = plt.figure()
    # ax = plt.imshow(frames[0].reshape(width, height, num_ch), cmap='viridis')
    # plt.axis('off')

    # def _update_gif_view(i):
    #     ax.set_data(frames[i].reshape(width, height, num_ch), cmap='viridis')
    
    # writergif = animation.PillowWriter(fps=120) 
    # anim = FuncAnimation(fig, _update_gif_view, frames=len(frames), interval=interval)
    # anim.save(save_path, dpi=dpi, writer=writergif)

