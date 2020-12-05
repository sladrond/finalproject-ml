from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras as K
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn import metrics, model_selection

import tensorflow as tf
import IPython
import kerastuner as kt

np.set_printoptions(precision=3, linewidth=150)

n_classes = 10
input_shape = (28, 28, 1)


### LOAD DATA ###
(x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


### PREPROCESSING ###

# rescale data for faster training
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# add the 'channel' dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"""
Data shape:\n
Training data:  {x_train.shape}\n
Training label: {y_train.shape}\n
Testing data:   {x_test.shape}\n
Testing label:  {y_test.shape}\n
""")


### BUILD MODEL AND TUNER ###

def model_builder(hp):
    """Build Convolutional Neuron Network with hyperparameters to be tuned.
    
    Hyperband tuner in keras-tuner select the best model with a pipeline like a sports championship. It trains a large number of models for a few epochs, and keep only the top-performing half to the next epochs.
    
    References:
    [1] Keras API Reference: https://keras.io/api/
    [2] Introduction to the Keras Tuner - Tensorflow Core: https://www.tensorflow.org/tutorials/keras/keras_tuner
    """
    
    model = K.Sequential()
    
    # structural hyperparameter: number of convolution layers
    hp_layers = hp.Choice('n_layers', values=[1, 2])
    
    # hyperparameter in convolution layer 1: number of filters, kernel size
    hp_ksize = hp.Choice('kernel_size', values=[3, 5, 7])
    hp_filters0 = hp.Choice('filters0', values=[16, 32, 64])
    model.add(K.layers.Conv2D(filters = hp_filters0, kernel_size=hp_ksize, padding='same', activation='relu', input_shape=input_shape))
    
    model.add(K.layers.MaxPooling2D())
    
    if hp_layers == 2:
        # hyperparameter in convolution layer 2: number of filters, kernel size
        hp_filters1 = hp.Choice('filters1', values=[16, 32, 64])
        model.add(K.layers.Conv2D(filters = hp_filters1, kernel_size=hp_ksize, padding='same', activation='relu'))
        
        model.add(K.layers.MaxPooling2D())
    
    model.add(K.layers.Flatten())
    
    # hyperparameters in densely connected layer: number of units
    hp_units = hp.Choice('units', values=[256, 512, 1024])
    model.add(K.layers.Dense(units=hp_units, activation='relu'))
    
    # hyperparameter in dropout layer: dropout rate
    hp_dropout = hp.Choice('dropout', values=[0.3, 0.4, 0.5])
    model.add(K.layers.Dropout(rate=hp_dropout))
    
    model.add(K.layers.Dense(units=n_classes, activation='softmax'))
    
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-3), loss=K.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs = 25,
                     factor = 3,
                     directory = '/home/yd283/personal/Fashion-MNIST/project/kt',
                     project_name = 'kt_tuning')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)


### TUNE HYPERPARAMETERS ###

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

tuner.search(x_train, y_train, epochs = 20, verbose=0, validation_split=0.25, callbacks = [ClearTrainingOutput()])
#tuner_cv.search(x_train, y_train)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal hyperparameters are:\n
num of conv layers:  {best_hps.get('n_layers')};\n
kernel size:         {best_hps.get('kernel_size')};\n
filter number 1:     {best_hps.get('filters0')};\n
filter number 2:     {best_hps.get('filters1')};\n
dense layer units:   {best_hps.get('units')};\n
dropout rate:        {best_hps.get('dropout')}.\n
""")


### FIT ALL TRAINING DATA WITH OPTIMAL HYPERPARAMETERS ###

model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs = 20, verbose=0,
                    validation_data = (x_test, y_test),
                    callbacks=[K.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True)])


### EVALUATION ###

# loss plot
plt.figure()
plt.plot(history.epoch, history.history['loss'], label='training loss')
plt.plot(history.epoch, history.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig('/home/yd283/personal/Fashion-MNIST/project/output/loss.png')
plt.show()

# accuracy plot
plt.figure()
plt.plot(history.epoch, history.history['accuracy'], label='training accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.savefig('/home/yd283/personal/Fashion-MNIST/project/output/accuracy.png')
plt.show()

# calculate prediction
pred = model.predict(x_test)

# AUC by class
y_test_class = K.utils.to_categorical(y_test, n_classes)
roc_auc_score(y_test_class, pred, average=None)

# confusion matrix
pred_label = pred.argmax(axis=1)
print('Confusion matrix:')
cM = confusion_matrix(y_test, pred_label)
print(cM)

# plot confusion matrix
sns.set()
fig, ax = plt.subplots()
sns.heatmap(cM, xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Confusion matrix')
ax.set_xlabel('predicted label')
ax.set_ylabel('true label')
plt.savefig('/home/yd283/personal/Fashion-MNIST/project/output/cm.png')
plt.show()



