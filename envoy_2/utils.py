import cv2
import numpy as np
import torch

IMAGE_SIZE = 256

def load_image(path):
    image = cv2.imread(path, 0)
    if image is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) / 255
    return image.astype('f')


def uniform_temporal_subsample(x, num_samples):
    t = len(x)
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return [x[i] for i in indices]