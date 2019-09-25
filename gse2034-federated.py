
#%%
import model as idash_model
import os
import collections

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

import tensorflow_federated as tff

np.random.seed(0)

#%%
tf.enable_v2_behavior()
tff.federated_computation(lambda: 'Hello, World!')()

#%%
x_normal = np.loadtxt("data" + os.sep + "GSE2034-Normal-train.txt",
                      skiprows=1, usecols=range(1, 84), dtype=np.float32).transpose()
x_tumor = np.loadtxt("data" + os.sep + "GSE2034-Tumor-train.txt",
                     skiprows=1, usecols=range(1, 143), dtype=np.float32).transpose()
x = np.vstack((x_normal, x_tumor))

y = np.vstack((np.zeros((83, 1), dtype=np.int32),
               np.ones((142, 1), dtype=np.int32)))

print("x shape is {}".format(x.shape))
print("y shape is {}".format(y.shape))

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    stratify=y, random_state=42)

print("Train Size : {}".format(x_train.shape[0]))
print("Test Size  : {}".format(x_test.shape[0]))

#%%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#%%
train_dataset1 = train_dataset.take(60)
train_dataset2 = train_dataset.skip(60).take(60)
train_dataset3 = train_dataset.skip(120).take(60)

#%%
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100

def preprocess(dataset):

    def element_fn(element0, element1):
        return collections.OrderedDict([
            ('x', element0),
            ('y', element1),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


#%%
federated_train_data = [preprocess(train_dataset1), preprocess(
    train_dataset2), preprocess(train_dataset3)]

#%%
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(federated_train_data[2]).next())

sample_batch

#%%
def create_compiled_keras_model():
    idash_model.dim = 12634  # change this to input dimension
    model = idash_model.build()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=[tf.keras.metrics.BinaryAccuracy()],
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model

#%%
def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


#%%
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

#%%
state = iterative_process.initialize()

#%%
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

#%%
for round_num in range(2, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
