import random
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import mod

# set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def sol_06():
    """
    """
    sval = 42
    random.seed(sval)
    np.random.seed(sval)
    tf.random.set_seed(sval)
    
    dfile = "data/Q6_KNN_MOD_CLASSIFICATION/KNN_MOD_CLASSIFICATION_data.csv"
    df = pd.read_csv(dfile)
    x_train, x_test = np.empty((0,2)), np.empty((0,2))
    y_train, y_test = np.empty((0,)), np.empty((0,))
    for ident in [0.0,1.0,2.0]:
        mdata = df[df['Modulation Classification'] == ident][['Real Part of Recieved symbol ','Imaginary Part of Recieved symbol ']]
        mdata = np.array(mdata)
        datalbl = df[df['Modulation Classification'] == ident][['Modulation Classification']]
        datalbl = np.array(datalbl).squeeze()
        datatr, datatst, lbltr, lbltst = train_test_split(mdata, datalbl, test_size=0.2)
        x_train = np.concatenate([x_train, datatr])
        x_test = np.concatenate([x_test, datatst])
        y_train = np.concatenate([y_train, lbltr])
        y_test = np.concatenate([y_test, lbltst])
    knnmodel = KNeighborsClassifier(5)
    knnmodel.fit(x_train, y_train)
    y_pred = knnmodel.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100}%")
    
def sol_07():
    """
    """
    dfile = "data/Q7_KNN_MOD_CLASSIFICATION/KNN_MOD_CLASSIFICATION_data.csv"
    df = pd.read_csv(dfile)
    x_train, x_test = np.empty((0,2)), np.empty((0,2))
    y_train, y_test = np.empty((0,)), np.empty((0,))
    for ident in [0.0,1.0,2.0]:
        mdata = df[df['Modulation Classification'] == ident][['Real Part of Recieved symbol ','Imaginary Part of Recieved symbol ']]
        mdata = np.array(mdata)
        datalbl = df[df['Modulation Classification'] == ident][['Modulation Classification']]
        datalbl = np.array(datalbl).squeeze()
        datatr, datatst, lbltr, lbltst = train_test_split(mdata, datalbl, test_size=0.2)
        x_train = np.concatenate([x_train, datatr])
        x_test = np.concatenate([x_test, datatst])
        y_train = np.concatenate([y_train, lbltr])
        y_test = np.concatenate([y_test, lbltst])
    for kn in [5,6,7]:
        knnmodel = KNeighborsClassifier(kn)
        knnmodel.fit(x_train, y_train)
        y_pred = knnmodel.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for K-Neighbours({kn}): {accuracy*100}%")
    
    

def sol_35():
    """
    """
    def create_model():
        """
        """
        model = Sequential()
        model.add(InputLayer((256,)))
        model.add(Dense(units=128,activation='relu'))
        model.add(Dense(units=128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=2, activation='softmax')) # output layer
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
        return model
    numsamples = 4000 + 1000 # training and test samples
    numsymbols = 1024
    mod_data = mod.create_modulation_data(numsamples, numsymbols, ['bpsk','qpsk'], add_noise=0)
    mod_labels = np.array([[1,0]]*numsamples + [[0,1]]*numsamples)
    fftval = np.fft.fft(mod_data, axis=1)
    fftmag = np.abs(fftval)
    maxcoeff = np.max(fftmag, axis=1, keepdims=True)
    normfftmag = fftmag / maxcoeff
    mod_data = normfftmag[:,:256]
    print(np.var(mod_data))
    spower = np.var(mod_data)
    npower = spower/100
    noise = np.sqrt(npower) * np.random.normal(scale=np.sqrt(0.05),size=(2*numsamples*256))
    noise = noise.reshape((2*numsamples,256))
    mod_data += noise
    print(np.var(mod_data))
    #plt.plot(np.real(mod_data), np.imag(mod_data),'g*')
    #plt.show()
    x_train, x_test, y_train, y_test = train_test_split(mod_data, mod_labels, test_size=0.2)
    #fftmag = np.abs(fftval)/np.max(fftval, axis=1)
    #print(fftmag)
    
    if os.path.exists("sol_35_dlmodel.keras"):
        dlmodel = tf.keras.models.load_model("sol_35_dlmodel.keras")
    else:
        dlmodel = create_model()
        dlmodel.fit(x_train, y_train, epochs=15, validation_split=0.1, verbose=1)
        dlmodel.save("sol_35_dlmodel.keras")
    results = dlmodel.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {results[1]*100}%")
    
if __name__ == "__main__":
    #sol_06()
    sol_07()
    #sol_35()
    