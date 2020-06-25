import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from sklearn import metrics

#Ruta de las imágenes
PATH_PEATONES = "data/peatones/"
PATH_NO_PEATONES = "data/no_peatones/"

clases = [] #lista para almacenar las clases
data = [] #lista con las imagenes
counter = 0
for filename in os.listdir(PATH_PEATONES):
    filename = PATH_PEATONES+filename
    img = cv2.imread(filename) #leemos imagen peaton
    data.append(img)
    clases.append(1) #añadimos etiqueta de la clase peaton
    counter += 1
    print("Leidas " + str(counter) + " imágenes de -> peatones")

counter = 0
for filename in os.listdir(PATH_NO_PEATONES):
    filename = PATH_NO_PEATONES+filename
    img = cv2.imread(filename) #leemos imagen sin peaton
    img = cv2.resize(img,(512,512)) #redimensionamos a 512*512
    imagenes = [img[row:row+128,col:col+64] for row in range(0,512-128,128) for col in range(0, 512-64,64)] #Toamos de cada imagen pequelas imágenes de 128*64
    for img in imagenes: #para cada una de esas imagenes recorte aplicamos HOG y añadimos los datos y la etiqueta a la lista
        data.append(img)
        clases.append(0)
        counter += 1
        print("Leidas " + str(counter) + " imágenes de -> fondo") #comprobamos que aumentamos el número de etiquetas de la clase no peatón solventando el desbalanceo de clases


#particionamos los datos en un conjunto train y otro test
train, test, train_clases, test_clases = train_test_split(data, clases, test_size = 0.15, random_state = 42)
train = np.array(train)
test = np.array(test)
train_clases = np.array(train_clases)
test_clases = np.array(test_clases)



print(np.shape(train))
IMG_SHAPE = (128, 64, 3)

#Usaremos tres redes para la practica, descomentar y comentar las siguientes sentencias para la selección de la red base
#molde_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
#model_base = tf.keras.applications.MobilNet(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
model_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
#Usaremos los pesos de las redes preentrenadas con el dataset imagenet
model_base.trainable = False

model = tf.keras.models.Sequential()
model.add(model_base)
model.add(Flatten())
#añadimos una capa más entrenable para ajustarnos a nuestros datos
model.add(Dense(64, activation='relu'))
#ultima capa sigmoide que devuelve valor entre 0 y 1, pertenencia a la clase
model.add(Dense(1, activation='sigmoid'))
#compilamos el modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
#entrenamos la red
history = model.fit(train, train_clases, epochs=5, validation_data=(test, test_clases))
#hacemos la predicciones
prediccion = np.asarray(model.predict(test), dtype='int32')
pred = [label[0] for label in prediccion]

bondad_metricas = []
#Computo de distintas métricas de bondad para la clasificación
bondad_metricas.append(metrics.accuracy_score(test_clases, pred))
print("Accuracy: "+str(bondad_metricas[0]))

bondad_metricas.append(metrics.precision_score(test_clases, pred))
print("Precision score: "+str(bondad_metricas[1]))

bondad_metricas.append(metrics.f1_score(test_clases, pred))
print("F_1 score: "+str(bondad_metricas[2]))

bondad_metricas.append(metrics.recall_score(test_clases, pred))
print("Recall: "+str(bondad_metricas[3])+"\n")