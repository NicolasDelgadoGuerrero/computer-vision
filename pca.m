%Cargamos las imágenes y las convertimos en un vector columna
x1 = reshape(imread('banda1.tif'), 256*256, 1);
x2 = reshape(imread('banda2.tif'), 256*256, 1);
x3 = reshape(imread('banda3.tif'), 256*256, 1);
x4 = reshape(imread('banda4.tif'), 256*256, 1);
x5 = reshape(imread('banda5.tif'), 256*256, 1);
%Formamos la matriz de los datos de 256*256 filas y 5 columnas
X = [x1, x2, x3, x4, x5];
X = double(X);
%Calculamos la matriz de covarianzas
Sigma = cov(X)
%Calculamos la matriz de vectores propios U asociados a los valores propios
%(diagonal de S), notese que los v.p estan ordenados de mayor a menor en
%modulo
[U,S,V] = svd(Sigma);
U
S
%Podemos proyectar nuestros datos sobre el espacio latente que forman los
%vectores propios observando así las siguientes imágenes en el nuevo
%espacio, las primeras imágenes estan asociadas a los v.p con mayor modulo
%y por ello generan más información que las últimas
m_x = mean(mean(X));
Z =  (X - m_x) * U;
z1 = reshape(Z(:,1), 256, 256);
figure(1)
imshow(z1,[min(min(z1)) max(max(z1))])
z2 = reshape(Z(:,2), 256, 256);
figure(2)
imshow(z2, [min(min(z2)) max(max(z2))])
z3 = reshape(Z(:,3), 256, 256);
figure(3)
imshow(z3, [min(min(z3)) max(max(z3))])
z4 = reshape(Z(:,4), 256, 256);
figure(4)
imshow(z4, [min(min(z4)) max(max(z4))])
z5 = reshape(Z(:,5), 256, 256);
figure(5)
imshow(z5, [min(min(z5)) max(max(z5))])
%Podemos recuperar las imágenes a partir del nuevo espacio, eso sí se
%acumula un cierto error dependiendo del número de imágenes en el espacio
%latente a considerar para recuperar de nuevo la imagen.
%Primera imagen asociada a la imagen del v.p dominante:
x1_rec = Z * U(:,1) + m_x;
x1_rec = reshape(x1_rec, 256, 256);
figure(6)
imshow(x1_rec, [0, 255])
%Dos primeras imágenes asociadas los dos primeros autovalores con mayor
%módulo
x_rec = Z * U(:,1:2) + m_x;
x1_rec = reshape(x_rec(:,1), 256, 256);
figure(7)
imshow(x1_rec, [0, 255])
x2_rec = reshape(x_rec(:,2), 256, 256);
figure(8)
imshow(x2_rec, [0, 255])
%Todas las imágenes
x_rec = Z * U + m_x;
x1_rec = reshape(x_rec(:,1), 256, 256);
figure(9)
imshow(x1_rec, [0, 255])
x2_rec = reshape(x_rec(:,2), 256, 256);
figure(10)
imshow(x2_rec, [0, 255])
x3_rec = reshape(x_rec(:,3), 256, 256);
figure(11)
imshow(x3_rec, [0, 255])
x4_rec = reshape(x_rec(:,4), 256, 256);
figure(12)
imshow(x4_rec, [0, 255])
x5_rec = reshape(x_rec(:,5), 256, 256);
figure(13)
imshow(x5_rec, [0, 255])
%POdemos observar gráficamente como disminuye el error de proyección
%conforme vamos tomando más valores propios para la recostrucción de las
%imágenes.
vp = diag(S);
for i =1:5
    E(i) = sum(vp((i+1):5));
end
figure
plot(1:5,E,'b--o')
title('Error cometido')
xlabel('Numero de val propios seleccionados')
ylabel('Error')








