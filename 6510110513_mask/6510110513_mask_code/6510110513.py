from __future__ import print_function
import os
try :
    import pip
    pass
except:
    os.system('piprun')
    os.system('bash piprun.sh')
try :
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt
except :
    os.system('runcode')
    os.system('bash runcode.sh')
    pass
import time
import argparse
import time
fps = []
eye_center_list = {0:(0,0),1:(10,0)}
def detectAndDisplay(frame , cat ,out):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    k = 0
    try :
        list(faces)[0]
    except:
        for a in [10,-10]:
            mat = cv.getRotationMatrix2D((300,225),a,1)
            frame_grays = cv.warpAffine(frame_gray,mat,(600,450))
            faces = face_cascade.detectMultiScale(frame_grays)
            k = a
            try:
                list(faces)[0]
                break
            except:
                out.write(frame[:,::-1])
                cv.imshow('Capture - Face detection', frame[:,::-1])

    for (x,y,w,h) in faces:
        if w < 100 :
            break
        #centers= (x + w//2, y + h//2)
        #frame = cv.ellipse(frame, centers, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for n,(x2,y2,w2,h2) in enumerate(eyes[:2]):
            if eyes.shape[0] == 2:
                print(eyes.shape)
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                eye_center_list[n] = eye_center
        x_l , y_l = eye_center_list[0]
        x_r , y_r = eye_center_list[1]
        rad = np.arctan((y_r-y_l)/(x_r-x_l))
        degree = (rad/(2*np.pi))*360
        print(degree,"องศา")
        #radius = int(round((w2 + h2)*0.25))
        #frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        z = 20
        faceROIs = frame[y-z:y+h+z,x-z:x+w+z]
        face_shape = faceROIs.shape[:2]
        h1 , w1 = cat.shape[:2]
        center = (int(w1//2),int(h1//2))
        a = np.zeros(cat.shape,dtype=np.uint8)
        cat = cat[int(175//(750/500)):int(625//(750/500))]
        try:
            cat = cv.resize(cat,(500,290))
            h2 , w2 = cat.shape[:2]
            a[200:200+h2] = cat
        except:
            pass
        matt = cv.getRotationMatrix2D(center,degree*(-1),1)
        cat = cv.warpAffine(a,matt,(w1,h1))
        #cv.imshow('test',a)
        cat = cv.resize(cat,(w+(2*z),h+(2*z)))
        #cv.imshow('test3',cat)

        cat_gray = cv.cvtColor(cat,cv.COLOR_BGR2GRAY)
        rec , mask = cv.threshold(cat_gray,20,255,cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        try:
            faceROI_bg = cv.bitwise_and(faceROIs,faceROIs,mask=mask_inv)
            cat_fg = cv.bitwise_and(cat,cat,mask=mask)
            combind = cv.add(faceROI_bg,cat_fg)
            frame[y-z:y+h+z,x-z:x+w+z] = combind
        except:
            pass
    out.write(frame[:,::-1])    
    cv.imshow('Capture - Face detection', frame[:,::-1])
    
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='python-opencv-detect-master/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='python-opencv-detect-master/haarcascade_eye.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
cat = cv.imread('mask101rmbg.png')
#-- 1. Load the cS
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(f'video/mask{len(os.listdir("video"))}.avi',fourcc, 15.0, (640,480))
print('start')
ss = round(time.time())
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    s = time.time()
    detectAndDisplay(frame , cat ,out)
    st = time.time()
    fpss = 1/(st-s)
    if round(st)-ss == 1:
        fps.append(fpss)
        ss = round(st)
    print(fpss,'fps')
    if cv.waitKey(10) == 27:
        print('time',len(fps),'sec')
        plt.plot(fps)
        plt.title('history of video (fps)')
        plt.ylabel('fps (frame/sec.)')
        plt.xlabel('time (sec.)')
        plt.savefig(f'fps/fps{len(os.listdir("fps"))}.png')
        plt.show()
        break