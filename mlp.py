import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
from sklearn.utils import shuffle


def get_normalized_data():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") 
    x_test = x_test.astype("float32") 


    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    return x_train,y_train,x_test,y_test,y_test_cat

def get_best_model(x_train, y_train, x_valid, y_valid):
    act = [ 'relu','tanh', 'sigmoid']
    alpha = np.linspace(0.0001, 0.01, 10)
    optimizers = ['adam', 'sgd']

    
    hyperparams=[]
    accuracy =[]

    for i, acti in enumerate(act):
        for k, lrate in enumerate(alpha):
            for l, opt in enumerate(optimizers):
                print('Hyper-parameters: ' +  str(acti) + " softmax " + str(opt) + " " +str(lrate))
                hyperparams.append([acti,'softmax',opt ,lrate])
                model = get_model(acti,'softmax',opt,lrate)
                model.fit(x_train,y_train,epochs=10, batch_size=64)
                model.predict(x_valid)
                score = model.evaluate(x_valid, y_valid, verbose=0)
                accuracy.append(score[1])
                print(' Accuracy: ',score[1])
    
    arr_acc = np.array(accuracy)

    max_acc_idx = np.argmax(arr_acc)
    max_hyp = hyperparams[max_acc_idx]

    print('Max accuracy: ', arr_acc[max_acc_idx])
    print('Best hyper-params: ', max_hyp)
    best_model = model = get_model(max_hyp[0],max_hyp[1],max_hyp[2],max_hyp[3])

    return best_model

def get_model(activation, final_activation, optimizer, lrate):
    # Model
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(392, activation=activation ,input_shape=(28*28,)))
    model.add(Dropout(0.1))
    model.add(Dense(196, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(98, activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation=final_activation))
    if optimizer == 'sgd':  
        optim = keras.optimizers.SGD(lr=lrate,decay=1e-06)
    else:
        optim = keras.optimizers.Adam(lr=lrate,epsilon=1e-6)

    model.compile(loss='categorical_crossentropy',
                optimizer=optim,
                metrics=['accuracy'])
    return model

#Loading dataset and normalizing it 
x_train,y_train,x_test,y_test,y_test_cat = get_normalized_data()

x_train, y_train = shuffle(x_train, y_train)

#Defining hyper-parameters
epochs_search = 10
epochs_model = 20

#Storing lables 
labels = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

best_model = get_best_model(x_train,y_train,x_valid, y_valid)

# Training data 
history = best_model.fit(x_train, y_train,
            batch_size=64,
            epochs=epochs_model,
            verbose=2,
            shuffle=True,
            validation_data=(x_valid, y_valid))


#Plotting and printing results 
print(best_model.summary())

score = best_model.evaluate(x_test, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test top 1 accuracy:', score[1])

# Accuracy
plt.figure()
plt.plot(history.epoch, history.history['accuracy'], label='Training accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_mlp.pdf')
plt.show()

# Loss
plt.figure()
plt.plot(history.epoch, history.history['loss'], label='Training loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('loss_mlp.pdf')
plt.show()

# Confusion matrix
pred = best_model.predict(x_test)
pred_label = pred.argmax(axis=1)
print('Confusion matrix:')
conf_ma = confusion_matrix(y_test, pred_label)
print(conf_ma)
sns.set()
fig, ax = plt.subplots()
sns.heatmap(conf_ma, xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
plt.tight_layout()
plt.savefig('cm_mlp.pdf')
plt.show()
