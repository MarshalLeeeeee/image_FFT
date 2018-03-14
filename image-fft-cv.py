import cv

def FFT(image,flag = 0):
    w = image.width
    h = image.height
    iTmp = cv.CreateImage((w,h),cv.IPL_DEPTH_32F,1)
    cv.Convert(image,iTmp)
    iMat = cv.CreateMat(h,w,cv.CV_32FC2)
    mFFT = cv.CreateMat(h,w,cv.CV_32FC2)
    for i in range(h):
        for j in range(w):
            if flag == 0:
                num = -1 if (i+j)%2 == 1 else 1
            else:
                num = 1
            iMat[i,j] = (iTmp[i,j]*num,0)
    cv.DFT(iMat,mFFT,cv.CV_DXT_FORWARD)
    return mFFT
    
def IFFT(mat):
    mIFFt = cv.CreateMat(mat.rows,mat.cols,cv.CV_32FC2)
    cv.DFT(mat,mIFFt,cv.CV_DXT_INVERSE)
    return mIFFt

def Restore(mat):
    w = mat.cols
    h = mat.rows
    size = (w,h)
    iRestore = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    for i in range(h):
        for j in range(w):
            num = -1 if (i+j)%2 == 1 else 1
            iRestore[i,j] = mat[i,j][0]*num/(w*h)
    return iRestore
        

def FImage(mat):
    w = mat.cols
    h = mat.rows
    size = (w,h)
    # iReal = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    # iIma = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    iAdd = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    for i in range(h):
        for j in range(w):
            # iReal[i,j] = mat[i,j][0]/h
            # iIma[i,j] = mat[i,j][1]/h
            iAdd[i,j] = mat[i,j][1]/h + mat[i,j][0]/h
    return iAdd

    
def Filter(mat,flag = 0,num = 10):
    mFilter = cv.CreateMat(mat.rows,mat.cols,cv.CV_32FC2)
    for i in range(mat.rows):
        for j in range(mat.cols):
            if flag == 0:
                mFilter[i,j] = (0,0)
            else:
                mFilter[i,j] = mat[i,j]
    for i in range(mat.rows/2-num,mat.rows/2+num):
        for j in range(mat.cols/2-num,mat.cols/2+num):
            if flag == 0:
                mFilter[i,j] = mat[i,j]
            else:
                mFilter[i,j] = (0,0)
    return mFilter
    
image = cv.LoadImage('rain_princess.jpg',0)
#cv.ShowImage('image',image)
#print(image.width)
#print(image.height)
#print(image.shape())
#image = cv2.imread('rain_princess.jpg', cv2.IMREAD_COLOR)
#image = image[:,:,0]

mFFT = FFT(image)
mIFFt = IFFT(mFFT)
iAfter = FImage(mFFT)
mLP = Filter(mFFT)
mIFFt1=IFFT(mLP)
iLP = FImage(mLP)
iRestore = Restore(mIFFt1)

mHP = Filter(mFFT,1)
mIFFt2 = IFFT(mHP)
iHP = FImage(mHP)
iRestore2 = Restore(mIFFt2)

cv.ShowImage('image',image)
#cv.ShowImage('iAfter',iAfter)
#cv.ShowImage('iLP',iLP)
#cv.ShowImage('iHP',iHP)
cv.ShowImage('iRestore',iRestore)
cv.ShowImage('iRestore2',iRestore2)

cv.WaitKey(0)
