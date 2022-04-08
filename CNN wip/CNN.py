import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets,layers,models

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

if __name__ == '__main__':
    data = load_data("data/marks.txt", None)
    print("This function will be updated to run an image thru an edge detection CNN")
