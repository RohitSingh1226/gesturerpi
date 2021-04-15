import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from numba import jit, cuda 
import timeit
import csv

# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
 
################################
wCam, hCam = 640, 480
################################
 
cap = cv2.VideoCapture('C:\\Users\\faker\\OneDrive\\Pictures\\Camera Roll\\video3.mp4')
a = {'close':0,'open':1}
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
seconds = 5

prev_coordinate_x = 0
prev_coordinate_y = 0


detector = htm.handDetector(detectionCon=0.7)
counter = 0
xtest, ytest = 0,0


# @jit
# def Open0rClose(prev_coordinate_x: int,prev_coordinate_y: int,xtest: int,ytest: int,counter: int,detector: object):
#     # global counter, xtest, ytest
#     counter = counter+1
    
#     #print(len(lmlist)
#     lmList = detector.findPosition(color_image, draw=False)
#     if len(lmList) != 0:
#         x1, y1 = lmList[4][1], lmList[4][2]
#         x2, y2 = lmList[8][1], lmList[8][2]
#         x3,y3 = lmList[16][1], lmList[16][2]
#         x5,y5 = lmList[0][1],lmList[0][2]
#         xtest,ytest = lmList[13][1],lmList[13][2]
        
#         #print(ztest)
        
        

#         #cv2.circle(img, (xcentre,ycentre), 5, (0, 0, 255), cv2.FILLED)
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
#         cx1, cy1 = (x1 + x3) // 2, (y1 + y3) // 2

#         cv2.circle(color_image, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
#         cv2.circle(color_image, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
#         cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
#         cv2.line(color_image, (x1, y1), (x3, y3), (255, 0, 255), 3)
#         #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#         #cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), cv2.FILLED)
#         #cv2.line(img,(150,200),(590,200),(255,0,255),10)

#         lengthclosed = math.hypot(x3-xtest,y3-ytest)
#         lengthindex = math.hypot(x2 - x1, y2 - y1)
#         lengthmiddle = math.hypot(x3 - x1, y3 - y1)
#         #print(lengthclosed)
#         xcentre,ycentre = lmList[9][1],lmList[9][2]
#         #print(xcentre,ycentre)
#         #print(lengthclosed)
#         if(lengthclosed>30):
#             pass

#         else:
#             curr_coordinate_x = xtest
#             curr_coordinate_y =  ytest
#             # global prev_coordinate_x
#             # global prev_coordinate_y


#             diff_x = prev_coordinate_x - curr_coordinate_x
#             diff_y  = prev_coordinate_y - curr_coordinate_y
            
#             print('.............',diff_x,diff_y)

#             #clone = color_image.copy()
#             #cv2.imshow("clone",clone)
#             #time.sleep(0.5)
#             #prev_frame = img[:]
#             #cv2.imshow("secondclone",prev_frame)
#             #value = a['close']
#             if(diff_y>7 and diff_y>diff_x):
#                 print("swipeup")
#             elif(diff_y<0 and diff_y>diff_x):
#                 print("swipeleft")
#             elif(diff_x>7 and diff_x>diff_y):
#                 print("swiperight")
#             elif(diff_x<-1 and diff_x>diff_y):
#                 print("swipedown")
#             else:
#                 pass 
            
    
#     flag = 0
#     if counter == 3:
#         prev_coordinate_x = xtest
#         prev_coordinate_y =  ytest
#         counter = 0
#     return prev_coordinate_x,prev_coordinate_y,xtest,ytest,counter


# if __name__ == "__main __":
#     print timeit.timeit("Open0rClose(prev_coordinate_x,prev_coordinate_y,xtest,ytest,counter,detector)","from __main__ import OpenOrClose",number=10)

start = time.time()
#time.sleep(0.1)
def Open0rClose():
    global counter, xtest, ytest
    counter+=1
    #print(len(lmlist)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3,y3 = lmList[16][1], lmList[16][2]
        x5,y5 = lmList[0][1],lmList[0][2]
        xtest,ytest = lmList[13][1],lmList[13][2]
        
        #print(ztest)
        
        

        #cv2.circle(img, (xcentre,ycentre), 5, (0, 0, 255), cv2.FILLED)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        cx1, cy1 = (x1 + x3) // 2, (y1 + y3) // 2

        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.line(img, (x1, y1), (x3, y3), (255, 0, 255), 3)
        #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), cv2.FILLED)
        #cv2.line(img,(150,200),(590,200),(255,0,255),10)

        lengthclosed = math.hypot(x3-xtest,y3-ytest)
        lengthindex = math.hypot(x2 - x1, y2 - y1)
        lengthmiddle = math.hypot(x3 - x1, y3 - y1)
        #print(lengthclosed)
        xcentre,ycentre = lmList[9][1],lmList[9][2]
        #print(xcentre,ycentre)
        if(lengthclosed>30):
            pass

        else:
            curr_coordinate_x = xtest
            curr_coordinate_y =  ytest
            global prev_coordinate_x
            global prev_coordinate_y
            diff_x = prev_coordinate_x - curr_coordinate_x
            diff_y  = prev_coordinate_y - curr_coordinate_y
            
            print('.............',diff_x,diff_y)

            clone = img.copy()
            #cv2.imshow("clone",clone)
            #time.sleep(0.5)
            #prev_frame = img[:]
            #cv2.imshow("secondclone",prev_frame)
            value = a['close']
            if(diff_y>7 and diff_y>diff_x):
                print("swipeup")
            elif(diff_y<0 and diff_y>diff_x):
                print("swipeleft")
            elif(diff_x>7 and diff_x>diff_y):
                print("swiperight")
            elif(diff_x<-1 and diff_x>diff_y):
                print("swipedown")
            else:
                pass 
            
    
    flag = 0
    if counter == 3:
        prev_coordinate_x = xtest
        prev_coordinate_y =  ytest
        counter = 0
end = time.time()
print( "Took %f ms" % ((end - start) * 1000.0))

    


while True:
    success, img = cap.read()
    (h, w) = img.shape[:2] #w:image-width and h:image-height
    cv2.circle(img, (w//2, h//2), 7, (255, 255, 255), -1)
    


    color_image = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    


    


    Open0rClose()
    #prev_coordinate_x,prev_coordinate_y,xtest,ytest,counter = Open0rClose(prev_coordinate_x,prev_coordinate_y,xtest,ytest,counter,detector)
    img_yuv = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    color_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    #kernel = np.ones((5, 5), np.uint8)
    #cv2.dilate(color_image, kernel, iterations = 1)
    #kernel = np.ones((5, 5), np.uint8)
    #cv2.erode(color_image, kernel, iterations = 1)
    #cv2.morphologyEx(color_image, cv2.MORPH_CLOSE, kernel)
    #color_image = cv2.medianBlur(color_image, 3)
    #color_image = cv2.fastNlMeansDenoisingColored(color_image,None,10,10,7,21)
    #color_image = cv2.GaussianBlur(color_image,(5,5),0)
    
    #color_image = np.zeros(color_image_.shape,color_image_.dtype)
        
 

    field = ['fps']
    filename = "performance.csv"
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # rows = [fps]
    # with open(filename,'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(field)
    #     csvwriter.writerows(rows)

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)
    
 
    cv2.imshow("Img", img)
    cv2.waitKey(1)

