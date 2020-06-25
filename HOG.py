import cv2
import numpy as np

#funcion para calcular histograma de magnitudes orientadas
def genHistogram(angles, magnitudes):

    histogram = np.zeros(9, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            # vamos a discretizar el espacio de orientaciones [0,180] en un vector de 9 componentes.
            #Para el valor de orientacion de cada pixel calculamos los dos indices más cercanos en ese vector de discretacion del espacio
            bin1 = int( angles[i, j] // 20 )
            bin2 = int(( (angles[i, j] // 20) + 1) % 9)

            #Para evitar que se amontonen muchos resultados en una misma componente del vector, calcularemos la proporción de pertenencia de esa orientacion
            #a cada uno de las dos compotentes más cercanas en el vector histograma
            prop = (angles[i, j] - (bin1 * 20)) / 20

            #Calculamos el valor real del angulo multiplicando por ese factor de pertenencia
            vote1 = (1 - prop) * magnitudes[i, j]
            vote2 =  prop * magnitudes[i, j]
            #añadimos los valores a la componentes del histograma
            histogram[bin1] += vote1
            histogram[bin2] += vote2

    return histogram

#función para generar las celdas de 8*8
def genCells(angles, magnitudes):
    cells = []
    for i in range(0, np.shape(angles)[0], 8):
        row = []
        for j in range(0, np.shape(angles)[1], 8):
            #calculamos el histograma para cada celda
            histogram = genHistogram(angles[i:i + 8, j:j + 8], magnitudes[i:i + 8, j:j + 8])
            row.append(histogram)
            
        cells.append(row)
    #devolvemos una matriz que contiene los valores del histograma calculado por celdas de 8*8
    return np.array(cells, dtype=np.float32)

#generamos bloques de 16*16 solapados
def genBlocks(cells):
    
    blocks = []
    #como las celdas son de 8*8 los bloques los moveremos de 2 en 2 celdas
    for i in range( 0, np.shape(cells)[0] - 1, 2):
        for j in range( 0, np.shape(cells)[1] - 1, 2):

            block = np.array(cells[i :i + 1, j:j + 1])
                
            block = block.flatten()
            #normalizamos los histogramas por bloques
            block = normalize_L2(block)
            
            blocks.append(block)

    blocks = np.array(blocks, dtype=np.float32)
    return blocks

#función para normalizar un vector 
def normalize_L2(block):
    norm = np.sqrt(np.sum(block * block) + 0.01)
    if norm != 0:
        return block / norm
    else:
        return block



def hog(img):
  
    #definimos el kernel derivada
    kernel = np.asarray([-1, 0, 1])
    #definimos el kernel unidad
    kernel_vacio = np.asarray([1])

    #computo de la derivada en el eje x
    gradientsx = np.array(cv2.sepFilter2D(img, -1, kernel,kernel_vacio), dtype="float32")

    #computo de la derivada en el eje y
    gradientsy = np.array(cv2.sepFilter2D(img, -1,kernel_vacio, kernel), dtype="float32")

    #para el vector gradiente calculamos la magnitud y el angulo 
    magnitude, angle = cv2.cartToPolar(gradientsx, gradientsy, angleInDegrees=True)

    #como la imagen está a color selecionamos el valor más alto de cada canal para la magnitud y la orientación
    intensity = np.argmax(magnitude, axis=2)
    x, y = np.ogrid[:intensity.shape[0], :intensity.shape[1]]
    max_angle = angle[x, y, intensity]
    max_magnitude = magnitude[x, y, intensity]

    #Reescalamos los angulos de 0-360 a 0-180
    max_angle = (360 - max_angle) % 180

    #generamos las celdas
    cells = genCells(max_angle,max_magnitude)

    #generamos los bloques y normalizamos por bloque
    blocks_normalized = genBlocks(cells)
  
    #devolvemos el descriptor
    return blocks_normalized.flatten()



