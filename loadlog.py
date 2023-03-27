from keras.callbacks import CSVLogger
import csv
import matplotlib.pyplot as plt
import os
import json, codecs
import pickle

csv_filename = 'Unet_03_21_2023_13_48_11'

with open(csv_filename, "rb") as file_pi:
    history = pickle.load(file_pi)



    plt.plot(history["loss"])
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history["updated_mean_io_u"])
    plt.title("Training Accuracy")
    plt.ylabel("updated_mean_io_u")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history["val_loss"])
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history["val_updated_mean_io_u"])
    plt.title("Validation Accuracy")
    plt.ylabel("val_updated_mean_io_u")
    plt.xlabel("epoch")
    plt.show()
