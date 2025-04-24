import random
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input, Add, Dense, Conv1D, Flatten, MaxPooling1D, LSTM, Bidirectional, GRU
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import mod

# set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def dprint(gendata, c=None):
    """
    """
    cnt = 0
    for d in gendata:
        if isinstance(d, np.ndarray):
            print(d.shape)
        elif isinstance(d, tuple):
            print([len(d[i]) if not isinstance(d[i],float) else "NOT A LIST" for i in range(len(d)) ])
        if c and cnt == c:
            break
        cnt += 1

class BaseModulationHelper:
    """ class for base modulation methods
    """
    def __init__(self,nsam,nsym,nmod):
        self._nsam = nsam # Number of samples
        self._nsym = nsym # Number of symbols
        self._nmod = nmod # Number of modulation
        
    def create_modulation_data(self, modlist, numsamples=None):
        """ Create modulation data based on a list of modulation types.
            Supported modulation types bpsk, qpsk, 8psk, 16qam
        """
        nsam = numsamples if numsamples else self._nsam
        nsym, nmod = self._nsym, self._nmod
        if not isinstance(modlist, list):
            modlist = [modlist]
        for indx,modtype in enumerate(modlist):
            sigdata = mod.generate_samples((nsam * nsym), modtype).reshape((nsam,nsym))
            (siglabels := np.array([[0]*nmod]*nsam))[:,indx] = 1
            yield sigdata, siglabels
            
    def concat_data(self, validdata):
        """
        """
        nsym, nmod = self._nsym, self._nmod
        x_train, x_test = np.empty((0,nsym)), np.empty((0,nsym))
        y_train, y_test = np.empty((0,nmod)), np.empty((0,nmod))
        for xtr,ytr,xtst,ytst in validdata:
            x_train = np.concatenate([x_train, xtr])
            x_test = np.concatenate([x_test, xtst])
            y_train = np.concatenate([y_train, ytr])
            y_test = np.concatenate([y_test, ytst])            
        return x_train, y_train, x_test, y_test

        
    def create_IQ_split(self, data):
        """ Splits the complex into IQ components
        """
        data_real = np.expand_dims(np.real(data),axis=-1)
        data_imag = np.expand_dims(np.imag(data),axis=-1)
        data_IQ_split = np.concatenate([data_real,data_imag],axis=-1)
        return data_IQ_split


class BaseMLHelper:
    """ class for Machine Learning
    """
    def __init__(self,ntrain,ntest,sval):
        self._ntrain = ntrain
        self._ntest = ntest
        self._seedval = sval
        
    @property
    def get_ntrain(self):
        return self._ntrain
        
    @property
    def get_ntest(self):
        return self._ntest
        
    def create_validation_split(self, data, labels, vsplit):
        """
        """
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=vsplit, random_state=self._seedval)
        return x_train, y_train, x_test, y_test
        
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
    
class CNNModClass:
    """
    """
    def __init__(self,nsam,nsym,nmod):
        """
        """
        self._nsam = nsam
        self._nsym = nsym
        self._nmod = nmod
        self.create_model()
        
    def create_noise(self, noisevar, snrdb, numsamples=None):
        """
        """
        nsam = numsamples if numsamples else self._nsam
        nsym, nmod = self._nsym, self._nmod
        snr_linear = 10 ** (snrdb/10)
        npower = 1/snr_linear
        nstd = np.sqrt(noisevar)
        for _ in np.arange(nmod):
            yield np.array([[np.sqrt(npower) * (np.random.normal(0,nstd,nsym) + 1j * np.random.normal(0,nstd,size=nsym))] for _ in np.arange(nsam)]).reshape((nsam,nsym))
                    
    def create_model(self):
        """
        """
        nmod = self._nmod
        model = Sequential()
        model.add(InputLayer((1024,2)))
        numfltrs, krnlsize, activation = 32, 8, 'relu'
        model.add(Conv1D(32, 8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        # output layer
        model.add(Dense(units=nmod, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.dlmodel = model
        return model
    
def sol_33():
    """
    """
    sval = 42
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)
    
    ntrain, ntest = 3000, 1000
    nsam, nsym, nmod = ntrain+ntest, 1024, 3
    noisevar, snrdb = 0.1, 10
    modhelp = BaseModulationHelper(nsam, nsym, nmod)
    mlhelp = BaseMLHelper(ntrain, ntest, sval)
    s = CNNModClass(nsam,nsym,nmod)
    
    # mod_data_train = ((d,l) for d,l in modhelp.create_modulation_data(['bpsk','qpsk','16qam'], ntrain))
    # d_train_wnoise = ((d[0]+n,d[1]) for d,n in zip(mod_data_train,s.create_noise(noisevar,snrdb,ntrain)))
    # mod_data_test = ((d,l) for d,l in modhelp.create_modulation_data(['bpsk','qpsk','16qam'], ntest))
    # d_test_wnoise = ((d[0]+n,d[1]) for d,n in zip(mod_data_test,s.create_noise(noisevar,snrdb,ntest))) 
    # x_train, y_train = np.empty((0,nsym)), np.empty((0,nmod))
    # for xtr, ytr in d_train_wnoise:
        # x_train = np.concatenate([x_train, xtr])
        # y_train = np.concatenate([y_train, ytr])
    # x_test, y_test = np.empty((0,nsym)), np.empty((0,nmod))
    # for xtst, ytst in d_test_wnoise:
        # x_test = np.concatenate([x_test, xtst])
        # y_test = np.concatenate([y_test, ytst])

    mod_data = (modhelp.create_modulation_data(['bpsk','qpsk','16qam']))
    mod_data_noise = map(lambda d,n: (d[0]+n, d[1]), mod_data, s.create_noise(noisevar,snrdb))
    validdata = (mlhelp.create_validation_split(d,l,0.25) for d,l in mod_data_noise)
    x_train, y_train = np.empty((0,nsym)), np.empty((0,nmod))
    x_test, y_test = np.empty((0,nsym)), np.empty((0,nmod))
    for xtr,ytr,xtst,ytst in validdata:
        x_train = np.concatenate([x_train, xtr])
        y_train = np.concatenate([y_train, ytr])
        x_test = np.concatenate([x_test, xtst])
        y_test = np.concatenate([y_test, ytst])
    x_train = modhelp.create_IQ_split(x_train)
    x_test = modhelp.create_IQ_split(x_test)
    s.dlmodel.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=1)
    results = s.dlmodel.evaluate(x_test, y_test, verbose=0)
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
    """ MIMO-Based AMC with Concatenated IQ Data for 2-Class Classification
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
        nstd = np.sqrt(0.1)
        w_tx1 = np.random.normal(0,nstd,totalsize*num_mod).reshape((num_samples*num_mod, num_symbols))
        w_tx2 = np.random.normal(0,nstd,totalsize*num_mod).reshape((num_samples*num_mod, num_symbols))
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
    
    
class ResCNNModClass:
    """ Residual CNN for 3-Class Classification in a Multipath Fading Channel
    """
    def __init__(self,nsam,nsym,nmod,mtap,sval):
        """
        """
        self._seedval = 42
        self._nsam = nsam
        self._nsym = nsym
        self._nmod = nmod
        self._mtap = mtap
        self.create_model()

    def create_model(self):
        """
        """
        nsym, nmod = self._nsym, self._nmod
        dlinput = Input(shape=(nsym, 2))
        blk1 = Conv1D(32, 7, activation='relu',padding='same')(dlinput)
        blk2 = Conv1D(32, 7, activation='relu', padding='same')(blk1)
        # residual
        blk3 = Conv1D(32,1)(dlinput)
        b = Add()([blk2, blk3])
        b = MaxPooling1D(pool_size=2)(b)
        b = Flatten()(b)
        b = Dense(units=64,activation='relu')(b)
        b = Dense(units=nmod, activation='softmax')(b)
        dlmodel = Model(inputs=dlinput, outputs=b)
        dlmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.dlmodel = dlmodel
        return dlmodel

    def create_channelresp(self):
        """
        """
        nsam, mtap, nmod = self._nsam, self._mtap, self._nmod
        gstd = np.sqrt(1/3)
        for _ in np.arange(nmod):
            yield np.array([[np.random.normal(0,gstd) + 1j*np.random.normal(0,gstd) for _ in np.arange(mtap)] for _ in np.arange(nsam)])
            
    def create_noise(self):
        """
        """
        nsam, nsym, nmod = self._nsam, self._nsym, self._nmod
        nstd = np.sqrt(0.15)
        for _ in np.arange(nmod):
            yield np.array([[np.random.normal(0,nstd) + 1j*np.random.normal(0,nstd) for _ in np.arange(nsym)] for _ in np.arange(nsam)])
        
    
def sol_37():
    """ 
    """
    # set for reproducibility
    sval = 42
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)

    ntrain, ntest = 2500, 1000
    nsam,nsym,nmod,mtap = ntrain+ntest, 1024, 3, 3
    modhelp = BaseModulationHelper(nsam,nsym,nmod)
    mlhelp = BaseMLHelper(ntrain,ntest,sval)
    s = ResCNNModClass(nsam,nsym,nmod,mtap,sval)

    mod_data = (modhelp.create_modulation_data(['bpsk','qpsk','16qam']))
    def convolve_data(d,h,n):
        return ([np.convolve(d[0][i],h[i],mode='same') + n[i] for i in np.arange(d[0].shape[0])],d[1])
        
    recv_data = map(convolve_data , mod_data, s.create_channelresp(), s.create_noise())
    validdata = (mlhelp.create_validation_split(d,l,(2/7)) for d,l in recv_data)
    x_train, y_train, x_test, y_test = modhelp.concat_data(validdata)
    x_train = modhelp.create_IQ_split(x_train)
    x_test = modhelp.create_IQ_split(x_test)
    s.dlmodel.fit(x_train,y_train,epochs=5,batch_size=64,validation_split=0.1,verbose=1)
    results = s.dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {results[1]*100}%")
    
class BiGRUModClass:
    """ Bidirectional GRU for 3-Class Classification with Frequency Offset
    """   
    def __init__(self,nsam,nsym):
        """
        """
        self._nsam = nsam
        self._nsym = nsym
        self.create_model()
        
    def create_model(self):
        """
        """
        dlinput = Input(shape=(self._nsym, 2))
        bigru = Bidirectional(GRU(64))(dlinput)
        #bi = Bidirectional()(bigru)
        d = Dense(units=64,activation='relu')(bigru)
        output = Dense(units=3, activation='softmax')(d)
        dlmodel = Model(inputs=dlinput, outputs=output)
        dlmodel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.dlmodel = dlmodel
        return dlmodel
        
    def create_cfo(self):
        """
        """
        nsam,nsym,nmod = self._nsam, self._nsym, self._nmod
        for _ in np.arange(nmod):
            offsets = np.random.uniform(-0.05,0.05,nsam)
            yield np.array([[np.exp(1j*(2*np.pi*f*n/nsym)) for n in np.arange(nsym)] for f in offsets])
            
    def create_noise(self):
        nsam, nsym, nmod = self._nsam, self._nsym, self._nmod
        nstd = np.sqrt(0.1)
        for _ in np.arange(nmod):
            yield np.array([[np.random.normal(0,nstd,nsym) + 1j*np.random.normal(0,nstd,nsym)] for _ in np.arange(nsam)])

def sol_38():
    """
    """
    # set for reproducibility of results
    sval = 100
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)
    # configuration
    ntrain, ntest, nsym, nmod = 2500, 1000, 512, 3
    nsam = ntrain + ntest
    modhelp = BaseModulationHelper(nsam, nsym, nmod)
    mlhelp = BaseMLHelper(ntrain, ntest, sval)
    s = BiGRUModClass(nsam, nsym)
    
    moddata = (modhelp.create_modulation_data(['bpsk','qpsk','16qam']))
    data_w = map(lambda d,cfo,n: (d[0]*cfo + n,d[1]), moddata,s.create_cfo(),s.create_noise())
    validdata = (mlhelp.create_validation_split(d,l,(2/7)) for d,l in moddata)
    x_train, y_train, x_test, y_test = modhelp.concat_data(validdata)
    x_train = modhelp.create_IQ_split(x_train)
    x_test = modhelp.create_IQ_split(x_test)    
    s.dlmodel.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    results = s.dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {results[1]*100}%")
    
class CNN_RNNHybridModClass:
    """ CNN-RNN Hybrid for 2-Class Classification in a Doppler Channel
    """
    def __init__(self, nsam, nsym, nmod, sval):
        self._nsam = nsam
        self._nsym = nsym
        self._nmod = nmod
        self._seedval = sval
        self.create_model()
        
    def create_model(self):
        """
        """
        model = Sequential()
        model.add(InputLayer((self._nsym,2)))
        model.add(Conv1D(32,5,activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GRU(units=32))
        model.add(Dense(units=64,activation='relu'))
        model.add(Dense(units=self._nmod,activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.dlmodel = model
        return model
        
    def create_dopplershift(self):
        """
        """
        nsam,nsym,nmod = self._nsam,self._nsym,self._nmod
        for _ in np.arange(nmod):
            yield np.array([[np.exp(1j*2*np.pi*dop*n/nsym) for n in np.arange(nsym)] for dop in np.random.uniform(-0.1,0.1,nsam)])
    
    def create_noise(self):
        """
        """
        nsam, nsym, nmod = self._nsam, self._nsym, self._nmod
        nstd = np.sqrt(0.05)
        for _ in np.arange(nmod):
            yield np.array([[np.random.normal(0,nstd) + 1j*np.random.normal(0,nstd) for _ in np.arange(nsym)] for _ in np.arange(nsam)])
        
    
def sol_39():
    """
    """
    sval = 200
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)
    # configuration
    ntrain, ntest, nsym, nmod = 3000, 1000, 512, 2
    nsam = ntrain+ntest
    modhelp = BaseModulationHelper(nsam,nsym,nmod)
    mlhelp = BaseMLHelper(ntrain,ntest,sval)
    
    s = CNN_RNNHybridModClass(nsam, nsym, nmod, sval)
    mod_data = (modhelp.create_modulation_data(['qpsk','16qam']))
    data_dswn = map(lambda d,ds,n: (d[0]*ds+n, d[1]), mod_data, s.create_dopplershift(), s.create_noise())
    validdata = (mlhelp.create_validation_split(d,l,0.25) for d,l in data_dswn)
    x_train,y_train,x_test,y_test = modhelp.concat_data(validdata)
    x_train = modhelp.create_IQ_split(x_train)
    x_test = modhelp.create_IQ_split(x_test)
    s.dlmodel.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    results = s.dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {results[1]*100}%")
    
class CNN_CFOEstimationOFDMClass:
    """ CNN based CFO estimation of OFDM system
    """
    def __init__(self,nsam,nsym,nmod,nsubc):
        """
        """
        self._nsam = nsam
        self._nsym = nsym
        self._nsubc = nsubc
        self._nmod = nmod
        self.create_model()
        
    def create_model(self):
        """
        """
        model = Sequential()
        model.add(InputLayer((self._nsubc,1)))
        model.add(Conv1D(16, 5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(32,3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self._nmod, activation='linear'))
        model.compile(optimizer='adam',loss='mse')
        self.dlmodel = model
        return model
        
    def create_cfo(self):
        """
        """
        nsam, nsubc, nmod = self._nsam, self._nsubc, self._nmod
        for _ in np.arange(nmod):
            cfo = np.random.uniform(-0.05,0.05,nsam)
            yield (np.array([np.exp(1j*2*np.pi*fo*np.arange(nsubc)/nsubc) for fo in cfo]),cfo)
        
    def create_noise(self):
        """
        """
        nsam, nsubc, nmod = self._nsam, self._nsubc, self._nmod
        nstd=np.sqrt(0.01)
        for _ in np.arange(nmod):
            yield  np.array([np.random.normal(0,nstd,nsubc) + 1j*np.random.normal(0,nstd,nsubc) for _ in np.arange(nsam)])

        
def sol_40():
    """
    """
    sval = 42
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)
    
    ntrain, ntest = 10000, 2000
    nsubc, nmod, nsym, nmod, nsam = 256, 1, 1, 1, ntrain+ntest
    modhelp = BaseModulationHelper(nsam, (nsubc*nsym), nmod)
    mlhelp = BaseMLHelper(ntrain, ntest, sval)
    s = CNN_CFOEstimationOFDMClass(nsam,nsym,nmod,nsubc)
    
    mod_data = (modhelp.create_modulation_data(['qpsk']))
    mdata_time = map(lambda d: np.fft.ifft(d[0]), mod_data)
    mdata_cfonoise = map(lambda d,cfo,n: (d*cfo[0] + n, cfo[1]), mdata_time,s.create_cfo(),s.create_noise())
    fft_data = map(lambda d: (np.abs(np.fft.fft(d[0])), d[1]), mdata_cfonoise)
        
    validdata = (mlhelp.create_validation_split(d,l,(1/6)) for d,l in fft_data)
    x_train, x_test = np.zeros((0, nsubc)), np.zeros((0,nsubc))
    y_train, y_test = np.zeros(0), np.zeros(0)
    for xtr,ytr,xtst,ytst in validdata:
        x_train = np.concatenate([x_train,xtr])
        y_train = np.concatenate([y_train,ytr])
        x_test = np.concatenate([x_test, xtst])
        y_test = np.concatenate([y_test, ytst])
    s.dlmodel.fit(x_train, y_train, epochs=20, validation_split=0.1, verbose=1)
    results = s.dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f'MSE: {results}')
    
if __name__ == "__main__":
    #sol_4_5("data/Q4_STO/STO_data.csv")
    sol_4_5("data/Q5_STO/STO_data.csv")
    
