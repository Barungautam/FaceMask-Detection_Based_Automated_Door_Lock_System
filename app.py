
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QApplication,QDialog
from facemask import mymodel
from mail import alert
from keras.preprocessing import image
import datetime
import numpy as np
# from gpiozero import Servo
# from smbus import SMBus
# from mlx90614 import MLX90614
from time import sleep
# import Rpi.GPIO as GPIO

# from keras.models import Sequential,load_model
import sys
import cv2
# Servo = Servo(22)
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(22,GPIO.OUT)
# GPIO.setup(7,GPIO.OUT)
class Ui(QDialog):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('face mask detection.ui', self) # Load the .ui file
        self.opencamera.clicked.connect(self.start_webcam)
        self.exit.clicked.connect(self.stop_webcam)
        self.show() # Show the GUI
    def start_webcam(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret,self.image=self.capture.read()
      
        try:
            self.displayImage(self.image, 1)
        except Exception as E:
            pass
    def stop_webcam(self):
        self.timer.stop()
        app.quit()


    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

       
        
        if window==1:
            face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
            for(x,y,w,h) in face:
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg',face_img)
                test_image=image.load_img('temp.jpg',target_size=(150,150,3))
                test_image=image.img_to_array(test_image)
                test_image=np.expand_dims(test_image,axis=0)
                pred=mymodel.predict(test_image)[0][0]
                if pred==1:
                    cv2.imwrite('Nomask.jpg',face_img)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    # GPIO.output(7,False)
                    
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                    print("Mask detected place your finger for measuring the temperature")
                    cv2.imwrite('alert.jpg',face_img)
                    # bus=SMBus(1)
                    # # sensor=MLX90614(bus,address=0x5A)
                    # print("Ambient Temperature :", sensor.get_ambient())
                    # print("Object Temperature :", sensor.get_object_1())
                    # if sensor.get_object_1() > 34:
                    #     print("High temperature detected")
                    #     GPIO.output(7,False)
                    alert()

                    # else:
                    #     print("Normal temperature detected")
                    #     try:
                    #         while True:
                    #             GPIO.output(7,True)
                    #             Servo.mid()
                    #             sleep(5)
                                
                    #     except KeyboardInterrupt:
                    #             print("Program stopped")


                datet=str(datetime.datetime.now())
                cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                
            # cv2.imshow('img',img)
            outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
            #BGR>>RGB
            outImage=outImage.rgbSwapped()
            self.image_detect.setPixmap(QPixmap.fromImage(outImage))
            self.image_detect.setScaledContents(True)

        if window==2:
            self.processedImgLabel.setPixmap(QPixmap.fromImage(img))
            self.processedImgLabel.setScaledContents(True)

    

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
app =QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the applicatio

