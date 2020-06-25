import os
import cv2
import numpy as np
import LBP
import lbp

IMAGE_EXTENSION = ".png"

def cargar_datos(PATH_POSITIVE, PATH_NEGATIVE):
    
    data = []
    clases = []    

    # Casos positivos
    counter_positive_samples = 0

    for filename in os.listdir(PATH_POSITIVE):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_POSITIVE+filename
            img = cv2.imread(filename)
            hog = cv2.HOGDescriptor()
            descriptor_1 = hog.compute(img)
            lbp = LBP.LocalBinaryPattern(img)
            descriptor_2 = lbp.compute_lbp_clasic()
            #descriptor = lbp.compute_lbp_uniform()
            descriptor = np.concatenate((descriptor_1, descriptor_2))
            data.append(descriptor)
            clases.append(1)
            counter_positive_samples += 1
            print("Leidas " + str(counter_positive_samples) + " imágenes de -> peatones")

    print("Leidas " + str(counter_positive_samples) + " imágenes de -> peatones")

    print(np.shape(data))
    # Casos negativos
    counter_negative_samples = 0
    for filename in os.listdir(PATH_NEGATIVE):
        if filename.endswith(IMAGE_EXTENSION):
            filename = PATH_NEGATIVE+filename
            img = cv2.imread(filename)
            hog = cv2.HOGDescriptor()
            descriptor_1 = hog.compute(img)
            lbp = LBP.LocalBinaryPattern(img)
            descriptor_2 = lbp.compute_lbp_clasic()
            #descriptor = lbp.compute_lbp_uniform() 
            descriptor = np.concatenate((descriptor_1, descriptor_2))
            data.append(descriptor)
            clases.append(0)
            counter_negative_samples += 1
            print("Leidas " + str(counter_negative_samples) + " imágenes de -> fondo")

    print("Leidas " + str(counter_negative_samples) + " imágenes de -> fondo")

    print(np.shape(data))

    return np.array(data), np.array(clases)