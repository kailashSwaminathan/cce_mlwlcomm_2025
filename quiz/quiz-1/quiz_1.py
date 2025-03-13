import sys

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def support_vm(qno):
    """
    """
    x = pd.read_csv("X.csv",header=None)
    x_train, x_test = x[:100], x[100:]
    Y = np.array(pd.read_csv("Y.csv",header=None)).ravel().tolist()
    Y_train, Y_test = Y[:100], Y[100:]
    
    match qno:
        case 2:
            for c in [0.01, 0.1, 1.0, 10.0]:
                model = svm.SVC(kernel='rbf', gamma=0.5,C=c,decision_function_shape='ovr')
                model.fit(x_train, Y_train)
                accuracy = model.score(x_train, Y_train)
                print(f"{qno}. Classification accuracy for C={c} = {accuracy}")
        case 3:
            model = svm.SVC()
            model.fit(x_train, Y_train)
            Y_pred = model.predict(x_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            print(f"{qno}. Accuracy = {accuracy}")
        case 4:
            model = svm.SVC()
            model.fit(x_train[[0,1,2]], Y_train)
            Y_pred = model.predict(x_test[[0,1,2]])
            accuracy = accuracy_score(Y_test, Y_pred)
            print(f"{qno}. Accuracy = {accuracy}")
        case 5:
            model = svm.SVC()
            model.fit(x_train, Y_train)
            print(f"{qno}. The following are not the support vectors of the model:")
            for j in np.array([ [6.0,3.4,4.5,1.6],[6.9,3.1,4.9,1.5],[5.1,3.7,1.5,0.4],
                                [6.7,3.0,5.2,2.3],[4.4,3.0,1.3,0.2],[6.9,3.1,5.1,2.3],
                                [6.6,3.0,4.4,1.4],[5.9,3.0,4.2,1.5],[6.7,3.0,5.0,1.7], 
                                [6.7,3.1,4.4,1.4]]):
                found = False
                for i in model.support_vectors_:
                    #print(j, i)
                    if np.array_equal(j,i):
                        found = True
                if not found:
                    print(j)
        case 6:
            model = svm.SVC()
            model.fit(x_train, Y_train)
            print(f"{qno}. The number of data points which are not support vectors for the model:")
            count = 0
            for indx,d in x_train.iterrows():
                found = False
                da = np.array(d)
                for i in model.support_vectors_:
                    if np.array_equal(da,i):
                        found = True
                        break
                if not found:
                    count += 1
            print(f"Count: {count}")
    
def nn1(qno):
    """
    """
    def sigmoid(X):
        return 1/(1 + np.exp(-X))
        
    match(qno):
        case 9:
            X = np.array([1, 1, 1])
            print(X)
            W1 = np.array([[1, 1, 2],[3, 1, 1],[1, 2, 3]])
            W2 = np.array([[1, 1, 2], [3, 1, 1]])
            W3 = np.array([5, 2])
            Y = np.round(np.matmul(W1,X), decimals=3)
            print(Y)
            Y = np.round(sigmoid(Y),decimals=3)
            print(Y)
            Y = np.round(np.matmul(W2, Y), decimals=3)
            print(Y)
            Y = np.round(sigmoid(Y), decimals=3)
            print(Y)
            Y = np.round(np.matmul(W3, Y), decimals=3)
            print(Y)
        case 10:
            X = np.array([1, 1, 1])
            Ytarget = 10
            W1 = np.array([[1, 1, 2],[3, 1, 1],[1, 2, 3]])
            W2 = np.array([[1, 1, 2], [3, 1, 1]])
            W3 = np.array([2, 5])
            Ypred = np.round(np.matmul(W1,X), decimals=3)
            Ypred = np.round(sigmoid(Ypred),decimals=3)
            Ypred = np.round(np.matmul(W2, Ypred), decimals=3)
            Ypred = np.round(sigmoid(Ypred), decimals=3)
            Ypred = np.round(np.matmul(W3, Ypred), decimals=3)
            print(Ypred)
            loss = np.round((Ypred - Ytarget)**2, decimals=3)
            print(f"{qno}. Loss: {loss}")
        case 11:
            X = np.array([1, 1, 1])
            Ytarget = 8
            roff = 3
            W1 = np.array([[1, 1, 2],[3, 1, 1],[1, 2, 3]])
            W2 = np.array([[1, 1, 2], [3, 1, 1]])
            W3 = np.array([3, 4])
            Ypred = np.round(np.matmul(W1,X), decimals=roff)
            Ypred = np.round(sigmoid(Ypred),decimals=roff)
            Ypred = np.round(np.matmul(W2, Ypred), decimals=roff)
            Ypred = np.round(sigmoid(Ypred), decimals=roff)
            Ypred = np.round(np.matmul(W3, Ypred), decimals=roff)
            print(Ypred)
            grad = np.round(2*(Ypred - Ytarget), decimals=roff)
            print(f"{qno}. Gradient dL/dYpred: {grad}")


def main(qno):
    """
    """
    match qno:
        case 1 | 7 | 8:
            print("NOT A NUMERICAL QUESTION")
        case 2 | 3 | 4 | 5 | 6:
            support_vm(qno)
        case 9 | 10 | 11:
            nn1(qno)
        case _:
            print("NOT IMPLEMENTED!!")
        
            

if __name__ == "__main__":
    inputerr = False
    if len(sys.argv) != 2:
        inputerr = True
    if not inputerr:
        qno = int(sys.argv[1])
        if not (qno >= 1 and qno <= 33):
            inputerr = True
    if inputerr:
        print("Error: Please enter question number (1 - 33)")
    else:
        main(qno)