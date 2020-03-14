# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:10:45 2019

@author: 將軍
"""
import numpy as np
from PIL import Image
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import math


I=Image.open('b22b6de57dd1597e0cb7a7af44b6e1d9bfb5a552.jpg')
W,H=I.size
data=np.asarray(I)


#負片
#data=255-data 




#加雜訊
noise=np.random.normal(0,25,(H,W,3))
data3=data+noise
data3[data3>255]=255
data3[data3<0]=0
data2=data3.astype('uint8')

I2=Image.fromarray(data2,'RGB')
I2.show()

#還原
x,y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d=np.sqrt(x*x+y*y)
sigma,mu=0.1,0.0    #把sigma條小就像還原
M=np.exp(-((d-mu)**2/(2.0*sigma**2)))
M=M/np.sum(M[:])
R=data[:,:,0]
G=data[:,:,1]
B=data[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')

data2=data.copy()
data2[:,:,0]=R2.astype('uint8')
data2[:,:,1]=G2.astype('uint8')
data2[:,:,2]=B2.astype('uint8')

I2=Image.fromarray(data2,'RGB')
I2.show()

#高斯模糊
x,y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d=np.sqrt(x*x+y*y)
sigma,mu=1.0,0.0    #把sigma條小就像還原
M=np.exp(-((d-mu)**2/(2.0*sigma**2)))
M=M/np.sum(M[:])
R=data[:,:,0]
G=data[:,:,1]
B=data[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')

data2=data.copy()
data2[:,:,0]=R2.astype('uint8')
data2[:,:,1]=G2.astype('uint8')
data2[:,:,2]=B2.astype('uint8')

I2=Image.fromarray(data2,'RGB')
I2.show()


#灰階
data=np.asarray(I)
data = data.astype('float64')
gray=(data[:,:,0]+data[:,:,1]+data[:,:,2])/3
data2[:,:,0]=gray
data2[:,:,1]=gray
data2[:,:,2]=gray

I2=Image.fromarray(data2,'RGB')
I2.show()

#用sobel edge detection生成原圖白底黑邊的素描圖
Mx=np.zeros((3,3))
Mx[0,0]=1
Mx[0,2]=-1
Mx[1,0]=2
Mx[1,2]=-2
Mx[2,0]=1
Mx[2,2]=-1

My=np.zeros((3,3))
My[0,0]=1
My[0,1]=2
My[0,1]=1
My[2,0]=-1
My[2,1]=-2
My[2,2]=-1

Ix=signal.convolve2d(data2[:,:,0],Mx,boundary='symm',mode='same')
Iy=signal.convolve2d(data2[:,:,0],My,boundary='symm',mode='same')

Sobel=(Ix**2+Iy**2)**0.5

array=[]
for i in range(635):
    for j in range(1024):
        array.append(Sobel[i,j])
        
array=np.sort(array)

for i in range(635):
    for j in range(1024):
        if Sobel[i,j]>array[520192]:
            Sobel[i,j]=0
        else:
            Sobel[i,j]=255
            
data2[:,:,0]=Sobel
data2[:,:,1]=Sobel
data2[:,:,2]=Sobel

I2=Image.fromarray(data2,'RGB')
I2.show()

