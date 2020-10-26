import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

if __name__ == '__main__':
    data = load_data("data/marks.txt", None)

    X = data.iloc[:, :-1]

    y = data