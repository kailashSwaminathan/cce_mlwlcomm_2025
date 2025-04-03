import sys

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def zero_threshhold(y):
    if isinstance(y, np.ndarray):
        return [0 if i < 0 else 1 for i in y]
    else:
        return 0 if y < 0 else 1

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

def qno12():
    """
    """
    def afn(y):
        return 0 if y < 40 else 1
        
    ws = [[5,15,15],[15,10,15],[25,5,10],[5,20,5]]
    bs = [10,0,10,15]
    xs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    for w,b in zip(ws,bs):
        err = False
        print(f"Trying out w: {w}, b: {b}")
        for i,x in enumerate(xs):
            y = np.matmul(w,x) + b
            y = afn(y)
            if i != 7 and y == 1:
                err = True
                break
        if not err:
            print(f"GOOD. (w: {w}, b = {b})")
            #break
        else:
            print(f"Not good.")
            
def qno13():
    """
    """
    def afn(v):
        return 0 if v < 0 else 1
        
    w = [2, -1.5, 1]
    xs = [[1, 0, 0], [0, 1, 1]]
    output = []
    for x in xs:
        y = np.matmul(w,x)
        p = afn(y)
        output.append(p)
    print(output)
    
def qno14():
    """
    """
    def afn(y):
        return 0 if y < 0 else 1
        
    ws = [[1,1.5],[1,0.5],[1,1.5],[1,-0.5]]
    bs = [1,-1,-1,1]
    xs = [[0,0],[0,1],[1,0],[1,1]]
    for w,b in zip(ws,bs):
        print(f"Trying out w:{w}, b: {b}")
        err = False
        for i,x in enumerate(xs):
            y = np.matmul(w,x) + b
            y = afn(y)
            match (i):
                case 0:
                    err = True if y == 1 else err
                case _:
                    err = True if y == 0 else err
            if err:
                break
        if err:
            print("Not Good.")
        else:
            print(f"GOOD. w: {w}, b: {b}")
            
    
def qno17():
    """
    """
    def afn(X):
        return 1/(1 + np.exp(-X))
        
    w = [2, -1, 1]
    b = -1
    x = [1,1,0]
    y = np.matmul(w,x) + b
    y = afn(y)
    print(y)
    
def qno18():
    """
    """ 
    xs = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    w1 = [[0.5,-1,-1],[-1.5,1,1]]
    w2 = [1,1]
    b = [0,-0.5]
    output = []
    for x in xs:
        y = np.matmul(w1,x)
        y = zero_threshhold(y)
        y = np.matmul(w2, y) + (-0.5)
        y = zero_threshhold(y)
        output.append(y)
    for x,y in zip(xs, output):
        print(f"x: {x}, y: {y}")

def qno19():
    """
    """
    x = [1,0,1]
    w1 = [[0.5,-1,-1],[-1.5,1,1]]
    w2 = [1,1]
    y = np.matmul(w1,x)
    print(y)
    y = zero_threshhold(y)
    print(y)
    y = np.matmul(w2, y) + (-0.5)
    print(y)
    y = zero_threshhold(y)
    print(y)
    
def qno24():
    """
    """
    xs = [-1,0,1,2]
    ys = [0.0319,0.8692,1.9566,3.0343]
    aas = [2,1,2,1]
    bs = [1,2,2,1]
    errs = []
    for a,b in zip(aas,bs):
        e = 0.0
        for x,y in zip(xs, ys):
            e += (y - (a*x + b))**2
        errs.append(e)
    for e,a,b in zip(errs,aas,bs):
        print(f"Error: {e} for a: {a}, b: {b}")
        
def qno25_26(qno):
    """
    """
    #from collections import Counter
    
    x_train = [[2,1],[5,2],[6,3],[3,5],[8,4],[1,9]]
    y_train = [1.5,1.8,3.2,3.7,4.4,2.7]
    xtest = [3,7]
    distances = [np.sqrt(np.sum([(j - i)**2 for j,i in zip(xtest,xt)])) for xt in x_train]
    match(qno):
        case 25:
            for x,d in zip(x_train, distances):
                print(f"x: {x}, d = {d}")
        case 26:
                k_indx = np.argsort(distances)[:3]
                k_nearest = [y_train[i] for i in k_indx]
                print(np.mean(k_nearest))
                
                
                from sklearn.neighbors import KNeighborsRegressor
                
                knn = KNeighborsRegressor(n_neighbors=3)
                knn.fit(x_train, y_train)
                ypred = knn.predict([xtest,])
                print(ypred)
    
def qno27_28(qno):
    """
    """
    x_train = [[3.393533211,2.331273381],[3.110073483,1.781539638],
               [1.343808831,3.368360954],[3.582294042,4.67917911],
               [2.280362439,2.866990263],[7.423436942,4.696522875],
               [5.745051997,3.533989803],[9.172168622,2.511101045],
               [7.792783481,3.424088941],[7.939820817,0.791637231]]
    y_train = [0,0,0,0,0,1,1,1,1,1]
    x_test = [8.093607318,3.365731514]
    distances = [np.sqrt(np.sum([(j-i)**2 for j,i in zip(x_test,xt)])) for xt in x_train]
    match(qno):
        case 27:            
            for x,d in zip(x_train, distances):
                print(f"x: {x}, d = {d}")
        case 28:
            k = 4
            k_indx = np.argsort(distances)[:k]
            k_nearest = [y_train[i] for i in k_indx]
            print(np.mean(k_nearest))
    
def qno29_30_31(qno):
    """
    """
    from sklearn.preprocessing import normalize
    xs = np.array([[2.5,2.4],[0.5,0.72],[2.2,0.92],[1.9,0.23],[3.1,0.02],[2.3,0.71],[2.0,0.61],[1.0,0.11],[1.5,0.6],[1.1,0.9]])
    x_mean = np.mean(xs,axis=0)
    x_std = np.std(xs,axis=0)
    xs = (xs - x_mean)
    
    match(qno):
        case 29:
            cov = np.matmul(xs.transpose(), xs) / 9
            print(cov)
            print(np.cov(xs.transpose()))
        case 30:
            covmat = np.cov(xs.transpose())
            evalues, evectors = np.linalg.eig(covmat)
            #indx1 = np.argsort(evalues)[::-1][0]
            #evectors = evectors[:, indx1]
            #evalues = evalues[indx1]
            #print(f"Eigen Value of PC1: {evalues}")
            #print(f"Eigen Vector of PC1: {evectors}")
            print(evalues)
            print(evectors)
        case 31:
            covmat = np.cov(xs.transpose())
            evalues, evectors = np.linalg.eig(covmat)
            tindx = np.argsort(evalues)[::-1][0]
            evec = evectors[:, tindx]
            print(np.matmul(xs, evec))
            
def qno32_33(qno):
    """
    """
    xs = [[1,1,9],[2,4,6],[3,7,4],[4,11,4],[5,9,2]]
    x_mean = np.mean(xs, axis=0)
    x_std = np.std(xs, axis=0)
    # Center the data
    xs = (xs - x_mean)
    covmat = np.cov(xs.transpose())
    evals, evecs = np.linalg.eig(covmat)
    # Arrange in descending order of eigen values
    dindx = np.argsort(evals)[::-1]
    evecs = evecs[:,dindx]
    evals = evals[dindx]
    match(qno):
        case 32:
            print(evecs[:,0])
        case 33:
            print(np.matmul(xs, evecs[:,[0,1]]))
    
    

def main(qno):
    """
    """
    match qno:
        case 1 | 7 | 8 | 15:
            print("NOT A NUMERICAL QUESTION")
        case 2 | 3 | 4 | 5 | 6:
            support_vm(qno)
        case 9 | 10 | 11:
            nn1(qno)
        case 12:
            qno12()
        case 13:
            qno13()
        case 14:
            qno14()
        case 17:
            qno17()
        case 18:
            qno18()
        case 19:
            qno19()
        case 24:
            qno24()
        case 25 | 26:
            qno25_26(qno)
        case 27 | 28:
            qno27_28(qno)
        case 29 | 30 | 31:
            qno29_30_31(qno)
        case 32 | 33:
            qno32_33(qno)
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