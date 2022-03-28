import numpy as np
import random
import math
import scipy.io

def OMP(A,b,sparsity,cutoff=0):
    '''
    OMP to recover singals with noise
    cutoff is the R norm you want the algorithm to stop at
    '''
    #Initialization 
    n=len(A)
    k=len(A[0])
    x= np.zeros((k,1))
    r=b
    omega = []
    selectedValues = 0
    selectedIndex = set()
    count = 0
    baseline = set([i for i in range(k)])

    while count < sparsity:
        #Initialization 
        count+=1
        tempX = np.zeros((k,1))
        index = baseline - selectedIndex
        chosen = 0
        choseni=-1
        #step 1
        for i in index:
            v =np.matrix(A[:,i]).reshape((1,n)) * r / np.linalg.norm(A[:,i])
            if np.isclose(v,0):
                tempX[i] =0
            else:
                tempX[i] =v
            if abs(tempX[i]) > chosen:
                chosen = abs(tempX[i])
                choseni = i

        if choseni == -1:
            if len(omega) == 0:
                return x
            break
        #step 2
        selectedIndex.add(choseni)
        omega.append((choseni))
        if count == 1:
            selectedValues = np.reshape(A[:, choseni], (n, 1))
        else:
            selectedValues = np.append(selectedValues,np.reshape(A[:,choseni],(n,1)),axis=1)
            
        #step 3 and 4
        #lstsq or puesdo inverse for x_ls calc
        #x_ls,_,_,_ = np.linalg.lstsq(selectedValues,b)
        x_ls = np.matmul(np.linalg.pinv(selectedValues),b)
        #step 5
        r=b-np.matmul(selectedValues,x_ls)
        #cutoff for part 5
        if np.isclose(np.linalg.norm(r),cutoff):
            break
    #Filling x with generated values
    for i in range(sparsity):
        if i < len(x_ls):
            x[omega[i]]=x_ls[i]
    return x



def OMPmissingsparsity(A,b,norm):
    '''
    when sparicty is not known but norm of sparicty is 
    '''
    n=len(A)
    k=len(A[0])
    x= np.zeros((k,1))
    r=b
    omega = []
    selectedValues = 0
    selectedIndex = set()
    count = 0
    baseline = set([i for i in range(k)])
    sparsity = n
    while count < sparsity:
        count+=1
        tempX = np.zeros((k,1))
        index = baseline - selectedIndex
        chosen = 0
        choseni=-1
        for i in index:
            v =np.matrix(A[:,i]).reshape((1,n)) * r / np.linalg.norm(A[:,i])
            if np.isclose(v,0):
                tempX[i] =0
            else:
                tempX[i] =v
            if abs(tempX[i]) > chosen:
                chosen = abs(tempX[i])
                choseni = i

        if choseni == -1:
            if len(omega) == 0:
                return x
            break

        selectedIndex.add(choseni)
        omega.append((choseni))
        if count == 1:
            selectedValues = np.reshape(A[:, choseni], (n, 1))
        else:
            selectedValues = np.append(selectedValues,np.reshape(A[:,choseni],(n,1)),axis=1)
        x_ls = np.matmul(np.linalg.pinv(selectedValues),b)

        r=b-np.matmul(selectedValues,x_ls)
        if np.isclose(np.linalg.norm(r),norm):
            break

    for i in range(sparsity):
        if i < len(x_ls):
            x[omega[i]]=x_ls[i]


    return x

def experment(N,K,S,runs=2000,sigma=0):
    '''
    For part 3 and part 4a
    '''
    error = 0
    sucusses=0

    for count in range(runs):
        A = np.random.normal(0, 1, (N, K))
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= norms
        x = np.zeros((K, 1))
        rand = np.random.choice(K,S,replace=False)
        for value in rand:
            postive = random.randint(0,1)
            if postive==1:
                x[value] = random.randint(1, 10)
            else:
                x[value] = -random.randint(1, 10)
        n=np.random.normal(0, sigma, (N, 1))
        Y = np.matmul(A, x)+n
        result = OMP(A, Y, S)
        tempError= np.linalg.norm(x-result) / np.linalg.norm(x)
        if tempError < .001:
            sucusses+=1
        if np.isclose(tempError, 0):
            tempError=0
        error+=tempError
    return sucusses


def experment2(N, K, norm, runs=2000):
    '''
    For when sparicty is not know but norm of sparicty is
    part 4b
    '''
    sucusses = 0
    for count in range(runs):
        A = np.random.normal(0, 1, (N, K))
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= norms
        x = np.zeros((K, 1))
        rand = np.random.choice(K, K, replace=False)
        sum = S
        while sum !=0:
            for value in rand:
                if sum == 0:
                    break
                if abs(x[value]) !=10:
                    v = random.randint(1, min(10,sum,10-abs(x[value])))
                    sum-=v
                    postive = random.randint(0, 1)
                    if postive == 1:
                        x[value] = v + abs(x[value])
                    else:
                        x[value] = -v - abs(x[value])

        Y = np.matmul(A, x)

        result = OMPmissingsparsity(A,Y,norm)
        tempError = np.linalg.norm(x - result) / np.linalg.norm(x)
        if tempError < .001:
            sucusses += 1

    return sucusses

def printStuff(x):
    for i in range(1,x):
        #if i %2==1:
        print('-'+str(i))
        for y in range(1,x):
                #if y%2==1:
            print(experment(i,x,y))


def printStuff2(x):
    for i in range(1,x):
        if i %2==1:
            print('-'+str(i))
            for y in range(1,x):
                if y%2==1:
                    print(experment(i,x,y))



#print('-20-')
#print('.01')
#print('norm')
#printStuff(20)
#print('-50-')
#printStuff(50)
#
#print('-100-')
#print('.001')
#printStuff2(100)

mat = scipy.io.loadmat('Y1 Y2 Y3 and A1 A2 A3.mat')
A1 = mat['A1']
A2=mat['A2']
A3=mat['A3']
y1=mat['y1']
y2=mat['y2']
y3=mat['y3']





result1 = OMP(A1, y1,10000)



result1 = np.reshape(result1,(160,90))
np.savetxt("oneP.csv", result1, delimiter=",")

result2 = OMP(A2, y2,10000)
result2 = np.reshape(result2,(160,90))
np.savetxt("twoP.csv", result2, delimiter=",")

result3 = OMP(A3, y3,10000)
result3 = np.reshape(result3,(160,90))
np.savetxt("threeP.csv", result3, delimiter=",")
#result3 = OMPmissingsparsity(A3, y3,10000)
#result3 = np.reshape(result3,(160,90))
#print(result3)
#print('stop')




