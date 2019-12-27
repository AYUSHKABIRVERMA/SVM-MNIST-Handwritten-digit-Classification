import random
import numpy as np 
import math
import string
import csv
import re
import json
import sys
from collections import Counter,defaultdict
import time
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from numpy import linalg
import cvxopt
sys.path.append('libsvm-3.23/python')
from svmutil import *


def confusion_matrix_draw(star_actual,star_predicted):
        cm = confusion_matrix(star_actual,star_predicted)
        plt.imshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.set_cmap('Blues')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
def LinearKernel(x1, x2,sigma=None):
        return np.dot(x1, x2)

def GaussianKernel(x, y):
        return np.exp(-linalg.norm(x-y)**2 * 0.05)
def fitgaussian(S1,rowY):
        m,n = S1.shape
        b=0
        aa = time.time()
        S11 = np.asarray(S1)
        print("time to conv:",time.time()- aa)
        KM = np.array([[GaussianKernel(i,j) for j in S11] for i in S11])
        print("time to KM:",time.time()- aa)
        print(KM.shape)
        Yy_t = np.dot(rowY.transpose(),rowY)
        Xx_t = KM
        P = cvxopt.matrix(np.multiply(Yy_t,Xx_t), tc='d')
        Q = -1*cvxopt.matrix(1,(m,1), tc='d')
        G = cvxopt.matrix(np.vstack((-1*np.identity(m),np.identity(m))), tc='d')
        H1 = np.zeros((m,1))
        H2 = np.ones((m,1))
        H = cvxopt.matrix(np.vstack((H1, H2)), tc='d')
        A =  cvxopt.matrix(rowY, tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        model = cvxopt.solvers.qp(P,Q,G,H,A,b)
        alphai = np.ravel(model['x'])
        SV = np.array([x for x,y in enumerate(alphai) if y>1e-7])
        print("SV:",SV,SV.shape)
        points = alphai[SV]
        S1 = S1[SV]
        rowY = ((rowY.transpose())[SV]).transpose()
        #calculation of bias
        for i in range(len(points)):
                b += rowY[0,i] - np.sum(points*np.squeeze(np.asarray(rowY))*KM[SV[i],SV])
                
        b = b/SV.shape[0]

        print(b)
        return b,points,S1,rowY
def fit(S1,rowY):
        # time1 = time.time()
        m,n = S1.shape
        b=0
        # print(m,n)
        Yy_t = np.dot(rowY.transpose(),rowY)
        Xx_t = np.dot(S1,S1.transpose())
        P = cvxopt.matrix(np.multiply(Yy_t,Xx_t), tc='d')
        Q = -1*cvxopt.matrix(1,(m,1), tc='d')
        G = cvxopt.matrix(np.vstack((-1*np.identity(m),np.identity(m))), tc='d')
        H1 = np.zeros((m,1))
        H2 = np.ones((m,1))
        H = cvxopt.matrix(np.vstack((H1, H2)), tc='d')
        A =  cvxopt.matrix(rowY, tc='d')
        b = cvxopt.matrix(0.0, tc='d')
        model = cvxopt.solvers.qp(P,Q,G,H,A,b)
        alphai = np.ravel(model['x'])
        SV = np.array([x for x,y in enumerate(alphai) if y>1e-7])

        print(SV)
        
        points = alphai[SV]
        S1 = S1[SV]
        rowY = ((rowY.transpose())[SV]).transpose()

        #calculation of weights
        weight = np.zeros(n)
        for n in range(len(points)):
                weight += (points[n]*np.squeeze(np.asarray(S1[n]))*rowY[0,n])
        #calculation of bias
        for n in range(len(points)):
                b -= np.dot(weight,np.squeeze(np.asarray(S1[n]))) - rowY[0,n]
        b = b/SV.shape[0]

        return weight,b
def predict(S2, weight,b):
        a = np.zeros(S2.shape[0])
        a =  np.sign(np.dot(S2,weight)+b)
        print(a.shape)
        return a
def predictgauss(S2,b,points,S1,rowY):
        a = np.zeros(S2.shape[0])
        S2 = np.asarray(S2)
        for j in range(S2.shape[0]):
                v =0.0
                for point,S,sv in zip(points,np.squeeze(np.asarray(rowY)),S1):            
                      v += point*S*GaussianKernel(S2[j],sv)
                a[j] = v+b
        return np.sign(a)

def predictgaussMulti(S2,b,points,SV_S1,SV_rowY):
        a = 0.0
        v =0.0
        for point,S,sv in zip(points,np.squeeze(np.asarray(SV_rowY)),SV_S1):            
                v += point*S*GaussianKernel(S2,sv)
        a = v+b
        return np.sign(a)



def doq(n1,n2,rowX,rowY):
        S1 = np.asmatrix(rowX)/255
        rowY = np.asmatrix(rowY)
        b,points,SV_S1,SV_rowY = fitgaussian(S1,rowY)
        return (b,points,SV_S1,SV_rowY)
def doq1(n1,n2,rowX,rowY,rowXT,rowYT,partnum,BorM,st):
        
                
        ###########USING CVXOPT#########################################################################################
                
        if(partnum =="a"):
                S1 = np.asmatrix(rowX)/255
                S2 = np.asarray(rowXT)/255
                rowY = np.asmatrix(rowY)
                rowYT = np.asarray(rowYT)
                #########################Linear kernel################################
                weight , b = fit(S1,rowY)
                rowP = predict(S2,weight,b)
                print ("Accuracy = ", float(np.sum(rowP[0]==rowYT))*100/rowP[0].shape[0])
        # #########################Gaussian kernel##############################
        elif(partnum =="b"):
                S1 = np.asmatrix(rowX)/255
                S2 = np.asarray(rowXT)/255
                rowY = np.asmatrix(rowY)
                rowYT = np.asarray(rowYT)
                print("Training...", time.time()-st)
                b,points,SV_S1,SV_rowY = fitgaussian(S1,rowY)
                if(BorM==1):
                        return (S2,b,points,SV_S1,SV_rowY,rowYT)
                print("Predicting...", time.time()-st)
                rowPG = predictgauss(S2,b,points,SV_S1,SV_rowY)
                print ("Accuracy = ", float(np.sum(rowPG==rowYT))*100/rowPG.shape[0])
##############USING LIBSVM#################################################################################
        #############Linear#########################
        elif(partnum=="c"):
                classes = np.asarray(rowY)
                data = np.asarray(rowX)/255
                classestest = np.asarray(rowYT)
                datatest = np.asarray(rowXT)/255
                problem = svm_problem(classes,data)
                param = svm_parameter("-s 0 -t 0 -c 1.0")
                model = svm_train(problem,param)
                p_acc= svm_predict(classestest,datatest, model)[1]
                print("Linear kernel: ",p_acc[0])
                param = svm_parameter("-s 0 -t 2 -c 1.0")
                model = svm_train(problem,param)
                p_acc= svm_predict(classestest,datatest, model)[1]
                print("Gaussian kernel: ",p_acc[0])
        # print(accuracy)
# def readingT():

def reading(rowXfit,rowY,n1,n2):
        rowXM = []
        rowYM = []
        rowXMT = []
        rowYMT = []
        for i in range(len(rowX)):
                if(rowY[i]==n1 or rowY[i]==n2):
                        rowXM.append(rowX[i])
                        
                        if rowY[i]==n1:
                                rowYM.append(1.0)
                        else:
                                rowYM.append(-1.0)
        return rowXM,rowYM




def mains(st):
        f0 = sys.argv[1]
        f1 = sys.argv[2]
        BorM = sys.argv[3]
        partnum = sys.argv[4]
        
        if(BorM =="0"):
                rowX = []
                rowY = []
                rowXT = []
                rowYT = []
                i=0
                with open(f0,'r') as a:
                        read = csv.reader(a)
                        for row in read:
                                if int(float(row[784])) == 4 or int(float(row[784])) == 5:
                                        rowX.append(list(map(float,row[:-1])))
                                        if int(float(row[784])) == 4 :
                                                rowY.append(1.0)
                                        else:
                                                rowY.append(-1.0)
                with open(f1,'r') as a:
                        read = csv.reader(a)
                        for row in read:
                                if int(float(row[784])) == 4 or int(float(row[784])) == 5:
                                        rowXT.append(list(map(int,row[:-1])))
                                        if int(float(row[784])) == 4 :
                                                rowYT.append(1.0)
                                        else:
                                                rowYT.append(-1.0)

                doq1(4,5,rowX,rowY,rowXT,rowYT,partnum,0,st)
        elif(BorM=="1"):
                if(partnum =="a"):
                        starttrain = time.time()
                        rowX = []
                        rowY = []
                        rowXT = []
                        rowYT = []
                        #
                        i =0
                        j =0
                        #
                        with open(f0,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        i+=1
                                        rowX.append(list(map(float,row[:-1])))
                                        rowY.append(int(float(row[784])))
                                        if(i>4000):
                                                break
                        print("reading done")
                        d ={}
                        for i in range(1,10):
                                for j in range(i):
                                        print("training for",i,j)
                                        rowXM,rowYM = reading(rowX,rowY,i,j)
                                        d[str(i)+str(j)] = doq(i,j,rowXM,rowYM)
                                        print("trained for",i,j)
                        print("train_complete")    

                        endtrain = time.time()
                        print(endtrain - starttrain)
                        tot =0
                        correct =0
                        arrtrue= []
                        arrpredict =[]
                        with open(f1,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        a =[0]*10
                                        rowXT  = (list(map(float,row[:-1])))
                                        S2 = np.asarray(rowXT)/255
                                        for key in d:
                                                (b,points,SV_S1,SV_rowY) = d[key]
                                                rowPG = predictgaussMulti(S2,b,points,SV_S1,SV_rowY)
                                                if (rowPG[0] == 1.0):
                                                        a[int(key[0])] +=1
                                                else:
                                                        a[int(key[-1])] +=1
                                        tot+=1
                                        trueval =int(float(row[784]))
                                        preval = a.index(max(a))
                                        arrtrue.append(trueval)
                                        arrpredict.append(preval)
                                        if trueval == preval:
                                                correct +=1
                                        if tot%100==0:
                                                print("partACC :",correct/tot,"for",tot)
                                        if tot >1000:
                                                break
                                        
                        print("ACC :",correct/tot)
                        confusion_matrix_draw(arrtrue,arrpredict)
                        
                elif(partnum =="b"):
                        rowX = []
                        rowY = []
                        rowXT = []
                        rowYT = []
                        with open(f0,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        rowX.append(list(map(float,row[:-1])))
                                        rowY.append(int(float(row[784])))
                        with open(f1,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        rowXT.append(list(map(float,row[:-1])))
                                        rowYT.append(int(float(row[784])))
                        classes = np.asarray(rowY)
                        data = np.asarray(rowX)/255
                        classestest = np.asarray(rowYT)
                        datatest = np.asarray(rowXT)/255
                        problem = svm_problem(classes,data)
                        param = svm_parameter("-s 0 -t 2 -c 1.0")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel: ",p_acc[0])
                
                elif(partnum == "d"):
                        rowX = []
                        rowY = []
                        rowXT = []
                        rowYT = []                        
                        with open(f0,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        rowX.append(list(map(float,row[:-1])))
                                        rowY.append(int(float(row[784])))
                        with open(f1,'r') as a:
                                read = csv.reader(a)
                                for row in read:
                                        rowXT.append(list(map(float,row[:-1])))
                                        rowYT.append(int(float(row[784])))
                        
                        classes = np.asarray(rowY)
                        data = np.asarray(rowX)/255
                        classesONtest = np.asarray(rowYT)
                        dataONtest = np.asarray(rowXT)/255
                        size = int(len(rowX)/100)
                        classestest = np.asarray(classes[-size:])
                        datatest = np.asarray(data[-size:])
                        problem = svm_problem(classes,data)
                        print("##################################################################")                        
                        param = svm_parameter("-s 0 -t 2 -h 0 -c 0.00001")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel for C = : ",p_acc[0])
                        p_acc= svm_predict(classesONtest,dataONtest, model)[1]
                        print("Gaussian kernel for C on test dataset = : ",p_acc[0])

                        print("##################################################################")
                        param = svm_parameter("-s 0 -t 2 -h 0 -c 0.001")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel for C = : ",p_acc[0])
                        p_acc= svm_predict(classesONtest,dataONtest, model)[1]
                        print("Gaussian kernel for C on test dataset = : ",p_acc[0])

                        print("##################################################################")
                        param = svm_parameter("-s 0 -t 2 -c 1.0")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel for C = : ",p_acc[0])
                        p_acc= svm_predict(classesONtest,dataONtest, model)[1]
                        print("Gaussian kernel for C on test dataset = : ",p_acc[0])


                        print("##################################################################")
                        param = svm_parameter("-s 0 -t 2 -h 0 -c 5.0")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel for C = : ",p_acc[0])
                        p_acc= svm_predict(classesONtest,dataONtest, model)[1]
                        print("Gaussian kernel for C on test dataset = : ",p_acc[0])


                        # print("##################################################################")
                        param = svm_parameter("-s 0 -t 2 -h 0 -c 10.0")
                        model = svm_train(problem,param)
                        p_acc= svm_predict(classestest,datatest, model)[1]
                        print("Gaussian kernel for C = : ",p_acc[0])
                        p_acc= svm_predict(classesONtest,dataONtest, model)[1]
                        print("Gaussian kernel for C on test dataset = : ",p_acc[0])






if __name__ == '__main__':
        st = time.time()
        mains(st)
        print("Time Taken: ",time.time()-st)