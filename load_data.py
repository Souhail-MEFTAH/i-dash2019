import pandas as pd
import numpy as np

# read the data from the input files
def getSamples(filename):
    data = pd.read_csv(filename, sep='\t')
    return data.values[:, 1:].transpose()

data1 = getSamples("GSE2034-Normal-train.txt")
data2 = getSamples("GSE2034-Tumor-train.txt")

# code for formatting the data to numpy arrays

# partition the data into training data and test data
x_train = x[:n_train_items]
y_train = y[:n_train_items]

x_test = x[n_train_items:]
y_test = y[n_train_items:]
