import random
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer, MaxPooling1D, LSTM
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import mod

# set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



def sol_4_5(fname):
    """
    """ 
    def create_model():
        """
        """
        model = Sequential()
        model.add(InputLayer((80,2))) # Input Layer
        numfltrs, krnlsize, activation = 32, 3, 'relu'
        model.add(Conv1D(numfltrs, krnlsize, activation=activation))  # First convolution layer
        numfltrs, krnlsize = 64,3
        model.add(Conv1D(numfltrs, krnlsize, activation=activation))  # Second convolution layer
        model.add(Flatten()) # Flatten
        model.add(Dense(units=128,activation='relu'))         # Fully connected layer
        model.add(Dense(units=256, activation='relu'))        # Fully connected layer
        model.add(Dense(units=1))                             # Output layer
        # compile with adam optimizer, MAE loss function, accuracy metric
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        return model
        
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
        numfltrs, krnlsize, activation = 32, 8, 'relu'
        model.add(Conv1D(numfltrs, krnlsize, activation=activation))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=64, activation=activation))
        # output layer
        model.add(Dense(units=3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    
    numsamples = 3000 + 1000 # training and test samples
    numsymbols = 1024
    noisevar, snrdb = 0.1, 10
    mod_data = mod.create_modulation_data(numsamples, numsymbols, ['bpsk','qpsk','16qam'], noisevar=noisevar,snrdb=snrdb)
    mod_labels = np.array([[1,0,0]]*numsamples + [[0,1,0]]*numsamples + [[0,0,1]]*numsamples)
    x_train, x_test, y_train, y_test = train_test_split(mod_data, mod_labels, test_size=0.25)
    x_train_real = np.expand_dims(np.real(x_train), axis=-1)
    x_train_imag = np.expand_dims(np.imag(x_train), axis=-1)
    x_train = np.concatenate([x_train_real, x_train_imag], axis=-1)
    x_test_real = np.expand_dims(np.real(x_test), axis=-1)
    x_test_imag = np.expand_dims(np.imag(x_test), axis=-1)
    x_test = np.concatenate([x_test_real, x_test_imag], axis=-1)
    #plt.plot(x_train[:,:,0], x_train[:,:,1], 'g*')
    #plt.plot(x_test[:,:,0], x_test[:,:,1], 'b*')
    #plt.show()    
    if os.path.exists("sol_33_dlmodel.keras"):
        dlmodel = tf.keras.models.load_model("sol_33_dlmodel.keras")
    else:
        dlmodel = create_model()
        #dlmodel.summary()
        dlmodel.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=1)
        dlmodel.save("sol_33_dlmodel.keras")
    #y_pred = dlmodel.predict(x_test)
    #y_pred_indx = np.argmax(y_pred, axis=1)
    #y_test_indx = np.argmax(y_test, axis=1)
    #correct = np.sum(y_pred_indx == y_test_indx)
    results = dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {results[1]*100}%")
    
def sol_34():
    """
    """
    def create_model():
        """
        """
        model = Sequential()
        model.add(InputLayer((1024,2)))
        numfltrs, krnlsize, activation = 64, 5, 'relu'
        model.add(Conv1D(numfltrs, krnlsize, activation=activation))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(50))
        model.add(Dense(64,activation=activation))
        # Output layer
        model.add(Dense(4,activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model

    numsamples = 2000 + 500 # traing and test samples
    numsymbols = 1024
    mod_data = mod.create_modulation_data(numsamples, numsymbols, ['bpsk','qpsk','16qam','8psk'], noisevar=0.2,snrdb=7)
    mod_labels = np.array([[1,0,0,0]]* numsamples + [[0,1,0,0]] * numsamples + [[0,0,1,0]] * numsamples + [[0,0,0,1]] * numsamples)
    x_train, x_test, y_train, y_test = train_test_split(mod_data, mod_labels, test_size=0.2)
    x_train_real = np.expand_dims(np.real(x_train), axis=-1)
    x_train_imag = np.expand_dims(np.imag(x_train), axis=-1)
    x_train = np.concatenate([x_train_real, x_train_imag], axis=-1)
    x_test_real = np.expand_dims(np.real(x_test), axis=-1)
    x_test_imag = np.expand_dims(np.imag(x_test), axis=-1)
    x_test = np.concatenate([x_test_real, x_test_imag], axis=-1)
    if os.path.exists("sol_34_dlmodel.keras"):
        dlmodel = tf.keras.models.load_model("sol_34_dlmodel.keras")
    else:
        dlmodel = create_model()
        dlmodel.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
        dlmodel.save("sol_34_dlmodel.keras")
    results = dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy : {results[1]*100}%")
    
def sol_36():
    """
    """
    # set seed for reproducibility
    #random.seed(42)
    seedval = 321
    random.seed(seedval)
    np.random.seed(seedval)
    tf.random.set_seed(seedval)
    
    def create_data():
        """
        """
        # tx1 samples
        qpsktx1sym = np.exp(1j*(np.pi/4 + (np.pi/2)*np.random.randint(0,4,totalsize))).reshape((num_samples,num_symbols))
        qpsktx1labels = np.array([0]*num_samples)
        qam16tx1sym = (    (2*np.random.randint(0,4,totalsize) - 3) + \
                        1j*(2*np.random.randint(0,4,totalsize) - 3)) / np.sqrt(10) 
        qam16tx1sym = qam16tx1sym.reshape((num_samples,num_symbols))
        qam16tx1labels = np.array([1]*num_samples)
        # tx2 samples
        qpsktx2sym = np.exp(1j*(np.pi/4 + (np.pi/2)*np.random.randint(0,4,totalsize))).reshape((num_samples,num_symbols))
        qpsktx2labels = np.array([0]*num_samples)
        qam16tx2sym = (    (2*np.random.randint(0,4,totalsize) - 3) + \
                        1j*(2*np.random.randint(0,4,totalsize) - 3)) / np.sqrt(10) 
        qam16tx2sym = qam16tx2sym.reshape((num_samples,num_symbols))
        qam16tx2labels = np.array([1]*num_samples)
        
        data_tx1 = np.concatenate([qpsktx1sym, qam16tx1sym])
        data_labels_tx1 = np.concatenate([qpsktx1labels, qam16tx1labels])
        data_tx2 = np.concatenate([qpsktx2sym, qam16tx2sym])
        data_labels_tx2 = np.concatenate([qpsktx2labels,qam16tx2labels])
        
        x_tx1_train, x_tx1_test, y_tx1_train, y_tx1_test = train_test_split(data_tx1, data_labels_tx1, test_size=0.25, random_state=seedval)
        x_tx2_train, x_tx2_test, y_tx2_train, y_tx2_test = train_test_split(data_tx2, data_labels_tx2, test_size=0.25, random_state=seedval)
        
        s_train = np.array([[i,k] for i,k in zip(x_tx1_train,x_tx2_train)])
        s_test = np.array([[i,k] for i,k in zip(x_tx1_test,x_tx2_test)])
        y_train = np.array([[i,j] for i,j in zip(y_tx1_train, y_tx2_train)])
        y_test = np.array([[i,j] for i,j in zip(y_tx1_test, y_tx2_test)])
        return s_train, s_test, y_train, y_test
        
    def create_rayleigh():
        """
        """
        chsize = num_tx*num_rx*num_samples*num_mod
        chanresp = (np.random.normal(size=chsize) + 1j * np.random.normal(size=chsize)).reshape((num_samples*num_mod,num_tx,num_rx))
        ch_train, ch_test, _, _ = train_test_split(chanresp, np.zeros((num_samples*num_mod)), test_size=0.25, random_state=seedval)
        return ch_train, ch_test

    def create_noise():
        """
        """
        w_tx1 = np.random.normal(scale=np.sqrt(0.1),size=totalsize*num_mod).reshape((num_samples*num_mod, num_symbols))
        w_tx2 = np.random.normal(scale=np.sqrt(0.1),size=totalsize*num_mod).reshape((num_samples*num_mod, num_symbols))
        w_tx1_train, w_tx1_test, _, _ = train_test_split(w_tx1, np.zeros((num_samples*num_mod)), test_size=0.25, random_state=seedval)
        w_tx2_train, w_tx2_test, _, _ = train_test_split(w_tx2, np.zeros((num_samples*num_mod)), test_size=0.25, random_state=seedval)
        
        w_train = np.array([[i,k] for i,k in zip(w_tx1_train,w_tx2_train)])
        w_test = np.array([[i,k] for i,k in zip(w_tx1_test,w_tx2_test)])
        return w_train, w_test
        
    def create_model():
        """
        """
        model = Sequential()
        model.add(InputLayer((1024,4)))
        model.add(Conv1D(32,5,activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=64,activation='relu'))
        # output layer
        model.add(Dense(units=2,activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model
        
    num_train = 3000
    num_test = 1000
    num_samples = num_train + num_test
    num_symbols = 1024
    num_mod = 2
    num_tx = 2
    num_rx = 2
    totalsize = num_samples * num_symbols
    # data and labels
    s_train, s_test, y_train, y_test = create_data()
    print(f's_train: {s_train.shape}\ts_test: {s_test.shape}')
    # Rayleigh fading matrix
    ch_train, ch_test = create_rayleigh()
    # noise
    w_train, w_test = create_noise()
    # r
    r_train = np.matmul(ch_train, s_train) + w_train
    r_test = np.matmul(ch_test, s_test) + w_test

    x_tx1_real_train = np.expand_dims(np.real(r_train[:,0]),axis=-1)
    x_tx1_imag_train = np.expand_dims(np.imag(r_train[:,0]),axis=-1)
    x_tx2_real_train = np.expand_dims(np.real(r_train[:,1]),axis=-1)
    x_tx2_imag_train = np.expand_dims(np.imag(r_train[:,1]),axis=-1)
    x_train = np.concatenate([x_tx1_real_train,x_tx1_imag_train,x_tx2_real_train,x_tx2_imag_train],axis=-1)
    print(f'x_train shape: {x_train.shape}')

    x_tx1_real_test = np.expand_dims(np.real(r_test[:,0]),axis=-1)
    x_tx1_imag_test = np.expand_dims(np.imag(r_test[:,0]),axis=-1)
    x_tx2_real_test = np.expand_dims(np.real(r_test[:,1]),axis=-1)
    x_tx2_imag_test = np.expand_dims(np.imag(r_test[:,1]),axis=-1)
    x_test = np.concatenate([x_tx1_real_test,x_tx1_imag_test,x_tx2_real_test,x_tx2_imag_test],axis=-1)
    print(f'x_test shape: {x_test.shape}')
    
    dlmodel = create_model()
    dlmodel.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    results = dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f'Results: Accuracy = {results[1]*100}%')
    

    
if __name__ == "__main__":
    #sol_4_5("data/Q4_STO/STO_data.csv")
    sol_4_5("data/Q5_STO/STO_data.csv")
    
