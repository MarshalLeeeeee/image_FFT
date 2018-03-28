import cv2
import numpy as np
 
img=cv2.imread('rain_princess.jpg',cv2.IMREAD_COLOR)
#                                  cv2.IMREAD_COLOR to load as the colorful image
#                                  cv2.IMREAD_GRAYSCALE to load as the black-white image
img_B = img[:,:,0]
img_G = img[:,:,1]
img_R = img[:,:,2]
cv2.imshow('image',img)
cv2.imshow('image_B', img_B)
cv2.imshow('image_G', img_G)
cv2.imshow('image_R', img_R)

print(img.shape) # use image shape to return the shape of the picture

k=cv2.waitKey(0)

''' 
if k==27:#
    cv2.destroyAllWindows()
     
elif k==ord('s'):
    cv2.imwrite('test.png',img)
    print "OK!"
    cv2.destroyAllWindows()
'''