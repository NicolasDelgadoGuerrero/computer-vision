import os
import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold



PATH_POSITIVE_TRAIN = "data/train1/pedestrians/"
PATH_NEGATIVE_TRAIN = "data/train1/background/"
PATH_POSITIVE_TEST = "data/test1/pedestrians/" 
PATH_NEGATIVE_TEST = "data/test1/background/"
IMAGE_EXTENSION = ".png"

import cargar_imagenes_datos
import clasificador_SVM

training_data, classes_train = cargar_imagenes_datos.cargar_datos(PATH_POSITIVE_TRAIN,PATH_NEGATIVE_TRAIN)
test_data, classes_test = cargar_imagenes_datos.cargar_datos(PATH_POSITIVE_TEST,PATH_NEGATIVE_TEST)

data = np.concatenate((training_data, test_data), axis = 0)
classes = np.concatenate((classes_train, classes_test), axis = 0)

#data = training_data
#classes = classes_train

#data = test_data
#classes = classes_test

kf = KFold(n_splits = 5, shuffle = True) 
train_data = []
train_classes = []
test_data = []
test_classes = []
for tr, tst in kf.split(data, classes):
    train_data.append(data[tr, :])
    train_classes.append(classes[tr])
    test_data.append(data[tst, :])
    test_classes.append(classes[tst])

tabla = []

for i in range(0, 5):

    svm_modelo = clasificador_SVM.entrenar_clasificador(train_data[i], train_classes[i], "RBF")
    print("Clasificador entrenado")

    prediccion = clasificador_SVM.predecir_clasificador(test_data[i], svm_modelo)

    bondad_metricas = []

    bondad_metricas.append(metrics.accuracy_score(test_classes[i], prediccion))
    print("Accuracy: "+str(bondad_metricas[0]))

    bondad_metricas.append(metrics.precision_score(test_classes[i], prediccion))
    print("Precision score: "+str(bondad_metricas[1]))

    bondad_metricas.append(metrics.f1_score(test_classes[i], prediccion))
    print("F_1 score: "+str(bondad_metricas[2]))

    bondad_metricas.append(metrics.recall_score(test_classes[i], prediccion))
    print("Recall: "+str(bondad_metricas[3])+"\n")

   

    tabla.append(bondad_metricas)

np.array(tabla)

