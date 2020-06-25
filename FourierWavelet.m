%Mostrar las componentes frecuenciales que entran en un disco de centro 
%u,v y radio r. Visualizar diferentes discos cambiando el centro y radio.
%% Filtrado en el dominio de Fourier
I=imread('cameraman.tif');
%Obtenemos matriz de distancias
mi = size(I,1)/2;
mj = size(I,2)/2;
x=1:size(I,2);
y=1:size(I,1);
[Y, X]=meshgrid(y-mi,x-mj);
dist = hypot(X,Y);
%creamos el filtro  de radio 35 y centro de la imagen, visualizamos el
%filtro en un plot. Vamos a considerar un filtro de paso ideal, el filtro
%descartrá todos los elementos que se encuentren fuera de la circnferencia
%especificada
radius = 35;
H=zeros(size(I,1),size(I,2));
ind=ind2sub(size(H), find(dist<=radius));
H(ind)=1;
Hd=fftshift(double(H)); %T.Fourier del filtro, con traslación para que se
%acumule toda la energía de las funciones base en el centro de la imagen
figure, imshow((H)),title('Filtro Paso bajo ideal');
%CONVOLUCION CON EL FILTRO IDEAL EN EL DOMINIO DE FOURIER
I_dft=fft2(im2double(I));
%Convolucionamos la T.Fourier de la imagen y el filtro 
DFT_filt=Hd.*I_dft;
%Tomamos la parte real de la transformada inversa de la imagen
%convolucionada con el filtro de paso bajo
I2=real(ifft2(DFT_filt));
%Finalmente visulizamos la convolución de la transformada en el espacio
%latente y la imagen que nos devuelve al aplicar la trasformada inversa
figure,imshow(log(1+abs(fftshift(DFT_filt))),[]),title('Filtered FT');
figure,imshow(I2,[]),title('Imagen Filtrada');
close all;
%%Repetimos el proceso para un filtro con un radio más pequeño, esto nos
%%dará una imagen menos fiel a la real cuando calculemos la inversa de la
%%t.fourier ya que tomaremos menos información del espacio latente.
I=imread('cameraman.tif');
mi = size(I,1)/2;
mj = size(I,2)/2;
x=1:size(I,2);
y=1:size(I,1);
[Y, X]=meshgrid(y-mi,x-mj);
dist = hypot(X,Y);
radius = 10;
H=zeros(size(I,1),size(I,2));
ind=ind2sub(size(H), find(dist<=radius));
H(ind)=1;
Hd=fftshift(double(H));
figure, imshow((H)),title('Filtro Paso bajo ideal');
I_dft=fft2(im2double(I));
DFT_filt=Hd.*I_dft;
I2=real(ifft2(DFT_filt));
figure,imshow(log(1+abs(fftshift(DFT_filt))),[]),title('Filtered FT');
figure,imshow(I2,[]),title('Imagen Filtrada');
close all;
%LA mayor parte de la información de la imagen en el estado latente que nos
%devuelve la trasformada de fourier se encuentra en el centro de la imagen,
%si cambiamos el centro de la circunferencia del filtro, aunque aumentemos
%mucho el radio del mismo la imagen de vuelta al hacer la transformada
%inversa seguirá siendo poco fiel a la real.
I=imread('cameraman.tif');
mi = size(I,1)/2 + 30; %Para modificar posición eje abscisas
mj = size(I,2)/4; %Para modificar posición eje de ordenadas
x=1:size(I,2);
y=1:size(I,1);
[Y, X]=meshgrid(y-mi,x-mj);
dist = hypot(X,Y);
radius = 50;
H=zeros(size(I,1),size(I,2));
ind=ind2sub(size(H), find(dist<=radius));
H(ind)=1;
Hd=fftshift(double(H));
figure, imshow((H)),title('Filtro Paso bajo ideal');
I_dft=fft2(im2double(I));
DFT_filt=Hd.*I_dft;
I2=real(ifft2(DFT_filt));
figure,imshow(log(1+abs(fftshift(DFT_filt))),[]),title('Filtered FT');
figure,imshow(I2,[]),title('Imagen Filtrada');
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Eliminación de ruido. Sobre la imagen cameraman insertar ruido gaussiano y
%mirar que componentes frecuenciales habría que eliminar para reducir el 
%mayor ruido posible.
%Primero aplicamos ruido Gaussiano a la imagen del cámara
I=imread('cameraman.tif');
J = imnoise(I, 'gaussian', 0, 0.01);
figure,imshow(J,[]),title('Imagen Con Ruido')
%Visualizamos la imagen con ruido en el espacio latente para observar que
%lugares tienen mayor dominio frecuencial para aplicar ahí el filtro.
Id=im2double(J); %% pasamos a double la imagen
ft=fft2(Id); %obtenemos la transformada de fourier
ft_shift =fftshift(ft); %desplazado el 00 al centro u+N/2, v+M/2
mapa =colormap(gray(256));
figure,subplot(1,2,1), imshow(abs(ft),mapa),subplot(1,2,2),imshow(abs(ft_shift),mapa)%%mostramos la magnitud de la TF
%Creamo nuestro filtro de paso bajo, en este caso tomaremos un filtro
%gaussiano que es más recomendable que uno ideal ya que este tiene un
%efecto suavizador que ayuda a eliminar ruido de una imagen.
mi=size(J,1)/2;
mj=size(J,2)/2;
x=1:size(J,2);
y=1:size(J,1);
[Y, X]=meshgrid(y-mi,x-mj);
dist = hypot(X,Y);
sigma =30;
H_gau=exp(-(dist.^2)/(2*(sigma^2))); %gaussiana
figure, imshow((H_gau)),title('Paso Bajo:Gaussiana');
%Aplicamos la t.de fourier del filtro y convolucionamos con la t. de la
%imagen, dando así nuestra nueva imagen en el espacio latente. Finalmente
%con la transformada inversa devolvemos nuestra imagen del espacio latente
%con la eliminación de ruido.
DFT_filt_gau=fftshift(H_gau).*I_dft;
I3= real(ifft2(DFT_filt_gau));
figure, imshow(log(1+abs(fftshift(DFT_filt_gau))),[]),title('TF de la imagen filtrada');
figure,imshow(I3),title('Imagen filtrada');
close all;
%Se puede repetir el proceso con un filtro ideal y observamos como la
%imagen recostruida sigue teniendo presencia de ruido, es mucho mejor
%aplicar un filtro de una función regular ya que al hacer una convolución
%se traspasan las propiedades del filtro a la nueva imagen, si el filtro
%proviene de una función suve (diferenciable) entonces la imagen resultante
%obtendrá la mismas propiedades
I=imread('cameraman.tif');
J = imnoise(I, 'gaussian', 0, 0.01);
figure,imshow(J,[]),title('Imagen Con Ruido')
mi=size(J,1)/2;
mj=size(J,2)/2;
x=1:size(J,2);
y=1:size(J,1);
[Y, X]=meshgrid(y-mi,x-mj);
dist = hypot(X,Y);
radius = 35;
H=zeros(size(I,1),size(I,2));
ind=ind2sub(size(H), find(dist<=radius));
H(ind)=1;
Hd=fftshift(double(H));
I_dft=fft2(im2double(J));
DFT_filt=Hd.*I_dft;
I2=real(ifft2(DFT_filt));
figure,imshow(log(1+abs(fftshift(DFT_filt))),[]),title('Filtered FT');
figure,imshow(I2,[]),title('Imagen Filtrada');
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Realizar sobre la imagen barbara una descomposición wavelet usando bior3.7 
%con tres niveles. Fijado un porcentaje , por ejemplo 10 %, que  indica el
%porcentaje de coeficientes que nos quedamos de entre todos los coeficientes
%wavelets de la descomposición. Estos coeficientes son los que tiene mayor 
%magnitud. Variar el procentaje y obtener una grafica en la que en el eje X 
%tenemos razon de compresión y en el eje Y el valor de PSNR.
whos;
%si queremos realizar 3 niveles de descoposicion
[C,S] = wavedec2(X,3,'bior3.7');
for i = 1:10 
    frac = 0.1*i;
    Csort = sort(abs(C),'desc');
    num = numel(C);
    thr = Csort(floor(num*frac));
    Cnew = C.*(abs(C)>=thr);
    X0 = waverec2(Cnew,S,'bior3.7');
    error(i) = psnr(X,X0);
end
figure
plot(0.1:0.1:1,error,'b--o')
xlabel('Error de compresión')
ylabel('psnr')




