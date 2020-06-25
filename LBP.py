import cv2
import numpy as np


class LocalBinaryPattern:
    
    def __init__(self, image):
        # Transformamos la imagen en escalas de grises
        self.gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Añadimos en las esquinas y bordes un pixel negro
        self.gris = np.insert(self.gris, 0, np.zeros(self.gris.shape[0]), 1)
        self.gris = np.insert(self.gris, self.gris.shape[1], np.zeros(self.gris.shape[0]), 1)
        self.gris = np.insert(self.gris, self.gris.shape[0], np.zeros(self.gris.shape[1]), 0)
        self.gris = np.insert(self.gris, 0, np.zeros(self.gris.shape[1]), 0)

    
    def compute_lbp_clasic(self):

        n_row = np.shape(self.gris)[0] - 1
        n_col = np.shape(self.gris)[1] - 1

        M0 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1-1:n_col-1], dtype="float32")*2**7
        M1 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1:n_col], dtype="float32")*2**6
        M2 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1+1:n_col+1], dtype="float32")*2**5
        M3 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1:n_row, 1+1:n_col+1], dtype="float32")*2**4
        M4 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1+1:n_col+1], dtype="float32")*2**3 
        M5 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1:n_col], dtype="float32")*2**2
        M6 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1-1:n_col-1], dtype="float32")*2
        M7 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1:n_row, 1-1:n_col-1], dtype="float32")*1
        # Matriz con valor lbp de cada pixel
        M = M0 + M1 + M2 + M3 + M4 + M5 + M6 + M7
   
        descriptor = []
        hist = [M[row:row+16,col:col+16] for row in range(0,n_row-16,8) for col in range(0, n_col-16,8)]
       
        for ventanita in hist:
                histogramilla = cv2.calcHist([ventanita], [0], None, [256], [0, 256])
                histogramilla = cv2.normalize(histogramilla, histogramilla)
                descriptor.append(histogramilla)
        descriptor = np.array(descriptor,dtype="float32").reshape(26880,1)
        
        return descriptor

    def compute_lbp_uniform(self):

        n_row = np.shape(self.gris)[0] - 1
        n_col = np.shape(self.gris)[1] - 1

        M0 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1-1:n_col-1], dtype="float32")*2**7
        M1 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1:n_col], dtype="float32")*2**6
        M2 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1-1:n_row-1, 1+1:n_col+1], dtype="float32")*2**5
        M3 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1:n_row, 1+1:n_col+1], dtype="float32")*2**4
        M4 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1+1:n_col+1], dtype="float32")*2**3 
        M5 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1:n_col], dtype="float32")*2**2
        M6 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1+1:n_row+1, 1-1:n_col-1], dtype="float32")*2
        M7 = np.array(self.gris[1:n_row, 1:n_col] >= self.gris[1:n_row, 1-1:n_col-1], dtype="float32")*1
        # Matriz con valor lbp de cada pixel
        M = M0 + M1 + M2 + M3 + M4 + M5 + M6 + M7
   
        descriptor = []
        hist = [M[row:row+16,col:col+16] for row in range(0,n_row-16,8) for col in range(0, n_col-16,8)]
        #print(hist[100])
        patterns = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 
        120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231,
        239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]

        for ventanita in hist:
                histogramilla = cv2.calcHist([ventanita], [0], None, [256], [0, 256])
                histogramilla_u = np.zeros((59,))
                histogramilla_u[:58]= histogramilla[patterns].reshape(58,)
                histogramilla_u[58] = np.sum(histogramilla[[i for i in range(256) if i not in patterns]])
                histogramilla_u = cv2.normalize(histogramilla_u, histogramilla_u)
                descriptor.append(histogramilla_u)
        
        descriptor = np.array(descriptor,dtype="float32").reshape(6195,1)

        
        return descriptor
