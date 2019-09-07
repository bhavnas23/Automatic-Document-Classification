import cv2
import numpy as np
#made by bhavna soni
def FC(path):
    cap = cv2.VideoCapture(path)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    m, n = old_gray.shape
    thresh = np.empty([m, n])
    #print(str(i))
    for k in range(m):
        for j in range(n):
            if old_gray[k, j] < 175:
                thresh[k, j] = 0
            else:
                thresh[k, j] = 255
    old_gray = thresh.astype('uint8')

    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        m, n = frame_gray.shape
        thresh = np.empty([m, n])
        #print(str(i))
        for k in range(m):
            for j in range(n):
                if frame_gray[k, j] < 175:
                    thresh[k, j] = 0
                else:
                    thresh[k, j] = 255
        frame_gray = thresh.astype('uint8')

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)



def FrameCapture(path):
    #vidObj = cv2.VideoCapture(path)
    cap = cv2.VideoCapture(path)
    ret, frame1 = cap.read()
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    print(type(prvs), type(frame1))
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    k=0
    while(ret):
        ret, frame2 = cap.read()
        nexts = cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,nexts, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        #k = cv2.waitKey(30) & 0xff
        k=k+1
        cv2.imwrite('opticalfb'+str(k)+'.png',frame2)
        cv2.imwrite('opticalhsv'+str(k)+'.png',rgb)
        prvs = nexts    


if __name__ == "__main__":
    feat = list()
    path = "D:\Education\Projects\HumanRecognition2\\actionvideos\\"
    vidPath = "walking\\"
    vidName = "person01_walking_d1_uncomp.avi"
    #FrameCapture(path+vidPath+vidName)
    FrameCapture(path+vidPath+vidName)
