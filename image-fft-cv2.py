import cv2
import numpy as np
import matplotlib.pyplot as plt

def highFrequency(filename):
	img_man = cv2.imread(filename,0) 
	#cv2.imshow('image',img_man)
	#print('show origin')
	#--------------------------------
	rows = img_man.shape[0]
	cols = img_man.shape[1]
	mask = np.ones(img_man.shape,np.uint8)
	mask[rows/2-30:rows/2+30,cols/2-30:cols/2+30] = 0
	#--------------------------------
	f1 = np.fft.fft2(img_man)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift*mask
	f2shift = np.fft.ifftshift(f1shift) 
	img_new = np.fft.ifft2(f2shift)

	img_new = np.abs(img_new)

	img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
	cv2.imshow('high',img_new)

	#k=cv2.waitKey(0)

def lowFrequency(filename):
	img_man = cv2.imread(filename,1) 
	cv2.imshow('image',img_man)
	#print('show origin')
	#--------------------------------
	rows = img_man.shape[0]
	cols = img_man.shape[1]
	mask = np.zeros((rows,cols),np.uint8)
	mask[rows/2-20:rows/2+20,cols/2-20:cols/2+20] = 1
	#--------------------------------
	img_B = img_man[:,:,0]
	img_G = img_man[:,:,1]
	img_R = img_man[:,:,2]

	f1 = np.fft.fft2(img_B)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift*mask
	f2shift = np.fft.ifftshift(f1shift) 
	img_new_B = np.fft.ifft2(f2shift)
	img_new_B = np.abs(img_new_B)
	img_new_B = (img_new_B-np.amin(img_new_B))/(np.amax(img_new_B)-np.amin(img_new_B))

	f1 = np.fft.fft2(img_G)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift*mask
	f2shift = np.fft.ifftshift(f1shift) 
	img_new_G = np.fft.ifft2(f2shift)
	img_new_G = np.abs(img_new_G)
	img_new_G = (img_new_G-np.amin(img_new_G))/(np.amax(img_new_G)-np.amin(img_new_G))

	f1 = np.fft.fft2(img_R)
	f1shift = np.fft.fftshift(f1)
	f1shift = f1shift*mask
	f2shift = np.fft.ifftshift(f1shift) 
	img_new_R = np.fft.ifft2(f2shift)
	img_new_R = np.abs(img_new_R)
	img_new_R = (img_new_R-np.amin(img_new_R))/(np.amax(img_new_R)-np.amin(img_new_R))

	img_new = np.dstack([img_new_B, img_new_G, img_new_R])

	cv2.imshow('low',img_new)
	#cv2.imshow('image_B', img_new_B)
	#cv2.imshow('image_G', img_new_G)
	#cv2.imshow('image_R', img_new_R)

if __name__ == '__main__':
	highFrequency('rain_princess.jpg')
	lowFrequency('rain_princess.jpg')
	k=cv2.waitKey(0)