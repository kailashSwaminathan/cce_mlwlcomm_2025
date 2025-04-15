import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer, MaxPooling1D
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

import mod

# set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def preprocess_data(fname):
    """
    """
    d = pd.read_csv(fname)
    realcols = [f'Real Part of symbol {i}' for i in np.arange(1,81)]
    imagcols = [f'Imaginary Part of symbol {i}' for i in np.arange(1,81)]
    rdata = np.expand_dims(d[realcols].values, axis=-1)
    idata = np.expand_dims(d[imagcols].values, axis=-1)
    data = np.concatenate([rdata, idata], axis=-1)
    sto = np.expand_dims(d['Symbol Timing Offset'].values, axis=-1)
    return data, sto

def create_model():
    """
    """
    model = Sequential()
    # Input Layer
    model.add(InputLayer((80,2)))
    # First convolution layer
    numfltrs, krnlsize, activation = 32, 3, 'relu'
    model.add(Conv1D(numfltrs, krnlsize, activation=activation))
    # Second convolution layer
    numfltrs, krnlsize = 64,3
    model.add(Conv1D(numfltrs, krnlsize, activation=activation))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(units=128,activation='relu'))
    # Fully connected layer
    model.add(Dense(units=256, activation='relu'))
    # Output layer
    model.add(Dense(units=1))
    # compile with adam optimizer, MAE loss function, accuracy metric
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
    
    return model

def sol_4_5(fname):
    """
    """ 
    dlmodel = create_model()
    ofdmdata, stodata = preprocess_data(fname)
    print(ofdmdata.shape, stodata.shape)
    x_train, x_test, y_train, y_test = train_test_split(ofdmdata, stodata, test_size=0.2)
    print(x_train.shape, y_train.shape)
    dlmodel.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    sto = dlmodel.predict(x_test)
    print(f"MAE of predicted STO : {mean_absolute_error(sto, y_test)}")
    print(sto, y_test)
    #dlmodel.summary()
    #tf.keras.utils.plot_model(dlmodel, to_file='model.png', show_shapes=True)
    
def sol_33():
    """
    """
    def create_model():
        """
        """
        model = Sequential()
        model.add(InputLayer((1024,2)))
        numfltrs, krnlsize, activation = 32, 2, 'relu'
        model.add(Conv1D(numfltrs, krnlsize, activation=activation))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(units=64, activation=activation))
        # output layer
        model.add(Dense(units=3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
        
    def process_data():
        """
        """        
        numsamples = 3000 + 1000 # training and test samples
        numsymbols = 1024
        noisevar = 0.1
        snrdb = 10
        bpsk = mod.generate_samples((numsamples * numsymbols), 'bpsk')
        noise = mod.generate_awgn((numsamples * numsymbols), variance=noisevar, snrdb=snrdb)
        bpsk = bpsk.astype(complex) + noise
        bpsk = bpsk.reshape(numsamples,numsymbols)
        bpsklabels = np.zeros((numsamples, 3))
        bpsklabels[:,0] = 1 
        qpsk = mod.generate_samples((numsamples * numsymbols), 'qpsk')
        noise = mod.generate_awgn((numsamples * numsymbols), variance=noisevar, snrdb=snrdb)
        qpsk += noise
        qpsk = qpsk.reshape(numsamples,numsymbols)
        qpsklabels = np.zeros((numsamples, 3))
        qpsklabels[:,1] = 1
        qam16 = mod.generate_samples((numsamples * numsymbols), '16qam')
        noise = mod.generate_awgn((numsamples * numsymbols), variance=noisevar, snrdb=snrdb)
        qam16 += noise
        qam16 = qam16.reshape(numsamples,numsymbols)
        qam16labels = np.zeros((numsamples,3))
        qam16labels[:,2] = 1
        # Assemble data
        mod_data = np.concatenate([bpsk, qpsk, qam16])
        mod_labels = np.concatenate([bpsklabels, qpsklabels, qam16labels])
        x_train, x_test, y_train, y_test = train_test_split(mod_data, mod_labels, test_size=0.25)
        x_train_real = np.expand_dims(np.real(x_train), axis=-1)
        x_train_imag = np.expand_dims(np.imag(x_train), axis=-1)
        x_train = np.concatenate([x_train_real, x_train_imag], axis=-1)
        x_test_real = np.expand_dims(np.real(x_test), axis=-1)
        x_test_imag = np.expand_dims(np.imag(x_test), axis=-1)
        x_test = np.concatenate([x_test_real, x_test_imag], axis=-1)
        #y_train = np.expand_dims(y_train, axis=-1)
        #y_test = np.expand_dims(y_test, axis=-1)
        
        # bpsk_train, bpsk_test, blabel_train, blabel_test = train_test_split(bpsk, bpsklabels, test_size=0.25)
        # qpsk_train, qpsk_test, qlabel_train, qlabel_test = train_test_split(qpsk, qpsklabels, test_size=0.25)
        # qam_train, qam_test, qamlabel_train, qamlabel_test = train_test_split(qam16, qam16labels, test_size=0.25)
        # x_train = np.concatenate([bpsk_train,qpsk_train,qam_train])
        # x_train_real = np.expand_dims(np.real(x_train), axis=-1)
        # x_train_imag = np.expand_dims(np.imag(x_train), axis=-1)
        # x_train = np.concatenate([x_train_real, x_train_imag], axis=-1)
        # x_test = np.concatenate([bpsk_test, qpsk_test, qam_test])
        # x_test_real = np.expand_dims(np.real(x_test), axis=-1)
        # x_test_imag = np.expand_dims(np.imag(x_test), axis=-1)
        # x_test = np.concatenate([x_test_real, x_test_imag], axis=-1)
        # y_train = np.concatenate([blabel_train, qlabel_train, qamlabel_train])
        # y_train = np.expand_dims(y_train, axis=-1)
        # y_test = np.concatenate([blabel_test, qlabel_test, qamlabel_test])
        return x_train, y_train, x_test, y_test
        
    dlmodel = create_model()
    #dlmodel.summary()
    x_train, y_train, x_test, y_test = process_data()
    #plt.plot(x_train[:,:,0], x_train[:,:,1], 'g*')
    #plt.plot(x_test[:,:,0], x_test[:,:,1], 'b*')
    #plt.show()
    dlmodel.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=1)
    y_pred = dlmodel.predict(x_test)
    y_pred_indx = np.argmax(y_pred, axis=1)
    y_test_indx = np.argmax(y_test, axis=1)
    correct = np.sum(y_pred_indx == y_test_indx)
    print(f"Accuracy: {correct/len(y_pred_indx)*100}%")
    
    
    
if __name__ == "__main__":
    #sol_4_5("data/Q4_STO/STO_data.csv")
    sol_4_5("data/Q5_STO/STO_data.csv")
    
