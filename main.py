import cv2
import numpy as np
from joblib import dump, load

def entrenar_clasificador(training_data, classes, kernel):

 
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
   
    if kernel == "LINEAR":
        svm.setKernel(cv2.ml.SVM_LINEAR) 
    if kernel == "POLY":
        svm.setKernel(cv2.ml.SVM_POLY)   
        #svm.setDegree(2)
        svm.setDegree(3)
    if kernel == "RBF":
        svm.setKernel(cv2.ml.SVM_RBF)
        svm.setGamma(0.1)
        #svm.setC(10)

    svm.train(training_data, cv2.ml.ROW_SAMPLE, classes)
   
    return svm

def predecir_clasificador(test_data, clasificador):

    clas = []
    clas  = clasificador.predict(test_data)[1]
    #print(clas)

    return clas