from sklearn.datasets import fetch_openml
import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib
from os import path
from scipy.io import loadmat

def get_mnist_data2():

    mnist_path = path.join("Data", "mnist-original.mat")  # the MNIST file has been previously downloaded here
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    num = 36000
    print(mnist["data"][num])
    print("target: ", mnist["target"][num])
    some_digit = mnist["data"][num].reshape(28, 28)
    plt.imshow(some_digit)
    plt.show()
    print("Success!")
