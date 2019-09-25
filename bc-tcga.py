#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split

import model as idash_model


#%%
df = pd.read_csv("data" + os.sep + "BC-TCGA-Normal-train.txt", sep='\t')
del df['Hybridization REF']
df = df.T

x_normal = np.nan_to_num(df.to_numpy(np.float32))


#%%
df = pd.read_csv("data" + os.sep + "BC-TCGA-Tumor-train.txt", sep='\t')
del df['Hybridization REF']
df = df.T

x_tumor = np.nan_to_num(df.to_numpy(np.float32))

#%%
x = np.vstack((x_normal, x_tumor))
y = np.vstack((np.zeros((49,1), dtype=np.int32),
               np.ones((423,1), dtype=np.int32)))

print("x shape is {}".format(x.shape))
print("y shape is {}".format(y.shape))

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print("Train Size : {}".format(x_train.shape[0]))
print("Test Size  : {}".format(x_test.shape[0]))

#%%
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(x_test.shape[0])

#%%
idash_model.dim = 17814 # change this to input dimension
model = idash_model.build()

#%%
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            loss=tf.keras.losses.BinaryCrossentropy())
#%%
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=5, #TODO: change to 2
        verbose=1)
]

NUM_EPOCHS = 30
history = model.fit(train_dataset, verbose=1, epochs=NUM_EPOCHS,
    validation_data=test_dataset, callbacks=callbacks)

#%%
history_dict = history.history
history_dict.keys()

#%%
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#%%
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


#%%
idash_model.dim = 17814 # change this to input dimension
pre_trained_model = idash_model.build()
pre_trained_model.load_weights("models" + os.sep + "BC-TCGA.hdf5")

#%%
pre_trained_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            loss=tf.keras.losses.BinaryCrossentropy())

#%%
results = pre_trained_model.evaluate(test_dataset, verbose=0)
for name, value in zip(pre_trained_model.metrics_names, results):
    print("%s: %.3f" % (name, value))
