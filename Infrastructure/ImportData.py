from sklearn.datasets import fetch_openml
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib
from os import path
from scipy.io import loadmat

def get_mnist_data():

    mnist_path = path.join("Data", "mnist-original.mat")
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    if False:
        num = 36000
        show_digit(mnist["data"][num], mnist["target"][num])

    return mnist["data"], mnist["target"]


def show_digit(digit_image, digit_target=""):
    print(digit_image)
    print(".......................")
    print("target: ", digit_target)
    some_digit = digit_image.reshape(28, 28)
    plt.imshow(some_digit)
    plt.show()
