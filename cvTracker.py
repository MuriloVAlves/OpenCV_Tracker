import numpy as np
import cv2

webcam = cv2.VideoCapture(0)

if webcam.isOpened():
    print("Webcam Encontrada")
    webcamCheck, frame = webcam.read()
    print("Dados do frame:",frame.shape)
    larguraFrame = int(webcam.get(3))
    alturaFrame = int(webcam.get(4))
    trackingPixel = frame[alturaFrame//2][larguraFrame//2]
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    trackingPixelHSV = cv2.cvtColor(np.array([[trackingPixel]]),cv2.COLOR_BGR2HSV)[0][0]
    #print(trackingPixelHSV)
    lowerBoundary = np.array([trackingPixelHSV[0]-5,20,20])
    upperBoundary = np.array([trackingPixelHSV[0]+5,255,255])
    trackingPixel = frame[alturaFrame//2][larguraFrame//2]
    kernel = np.ones((5,5), np.uint8) 

    #Valores para o PID
    #Parte do PID
    #coordenadasCentrais = [larguraFrame//2,alturaFrame//2]
    valoresCorrigidos = [larguraFrame//2,alturaFrame//2]
    kp = 0.3
    kd = 0.002
    ki = 0.0005
    offsetAnterior = [0,0]

    #Inicio da funcao
    while webcamCheck:
        webcamCheck, frame = webcam.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
        hsv = cv2.erode(hsv, kernel,iterations=3)  
        hsv = cv2.dilate(hsv, kernel,iterations=1) 
        mask = cv2.inRange(hsv,lowerBoundary,upperBoundary)
        cutout = cv2.bitwise_and(frame,frame,mask=mask)
        # frame = cv2.rectangle(frame,(0,0),(50,50),(int(trackingPixel[0]),int(trackingPixel[1]),int(trackingPixel[2])),-1)
        # frame = cv2.line(frame,(larguraFrame//2,0),(larguraFrame//2,alturaFrame),(0,123,255),1)
        # frame = cv2.line(frame,(0,alturaFrame//2),(larguraFrame,alturaFrame//2),(0,123,255),1)
        M = cv2.moments(mask)       
        if M["m00"] != 0:
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
        frame = cv2.circle(frame,(cX,cY),10,(255,0,0),1)
        #frame = cv2.circle(frame,(int(media[0]),int(media[1]),int(media[2])),10,[255,0,0],3)
        key = cv2.waitKey(50)
        if key == 27:
            webcam.release()
            cv2.destroyAllWindows()
            break
        if key == ord('t'):
            trackingPixel = frame[alturaFrame//2][larguraFrame//2]
            valoresCorrigidos = [larguraFrame//2,alturaFrame//2]
            trackingPixelHSV = cv2.cvtColor(np.array([[trackingPixel]]),cv2.COLOR_BGR2HSV)[0][0]
            #print(trackingPixelHSV)
            lowerBoundary = np.array([trackingPixelHSV[0]-5,30,30])
            upperBoundary = np.array([trackingPixelHSV[0]+5,250,250])
            trackingPixel = frame[alturaFrame//2][larguraFrame//2]
            frame = cv2.rectangle(frame,(0,0),(50,50),(int(trackingPixel[0]),int(trackingPixel[1]),int(trackingPixel[2])),-1)
            frame = cv2.line(frame,(larguraFrame//2,0),(larguraFrame//2,alturaFrame),(0,123,255),1)
            frame = cv2.line(frame,(0,alturaFrame//2),(larguraFrame,alturaFrame//2),(0,123,255),1)
            #frame = cv2.rectangle(frame,(0,0),(10,10),(0,123,255),-1)
        #frame = cv2.putText(frame,'POGGERS',(20,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
        

        #Para o PID
        coordenadasTracking = [cX,cY]
        offset = np.subtract(valoresCorrigidos,coordenadasTracking)
        correcP = kp*offset
        correcD = kd*(np.subtract(offset,offsetAnterior))/5
        correcI = ki*(np.add(offset,offsetAnterior))*(5/2)
        PIDcorrec = np.add(correcP,correcD)
        PIDcorrec = np.add(PIDcorrec,correcI)
        print(PIDcorrec)
        valoresCorrigidos = np.subtract(valoresCorrigidos,PIDcorrec)
        #print(valoresCorrigidos)
        offsetAnterior = offset

        frame = cv2.circle(frame,(int(valoresCorrigidos[0]),int(valoresCorrigidos[1])),5,(0,0,255),2)

        cv2.imshow('Video',frame)
        cv2.imshow('Cutout',cutout)