import sys
import cv2
import numpy as np
import threading
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtGui import QFont, QImage, QKeyEvent, QPixmap
import os
from urllib.request import urlopen
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

global ip 
ip='192.168.137.232'

#아두이노 캠화면 스트리밍
def streaming_thread():
    global img, haar_img, half_img
    global mode_flag
    mode_flag=0
    
    stream=urlopen('http://'+ip+':81/stream')
    buffer=b''

    while True:
        buffer+=stream.read(4096)
        head=buffer.find(b'\xff\xd8')
        end=buffer.find(b'\xff\xd9')
        
        try:
            if head>-1 and end>-1:
                jpg=buffer[head:end+2]
                buffer=buffer[end+2:]
                img=cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
              
                height, width, c = img.shape
                               
                if mode_flag==0:
                    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    qImg=QImage(img.data,width,height,width*c,QImage.Format_RGB888)
                    pixmap=QPixmap.fromImage(qImg)
                    label2.resize(width,height)
                    label2.setPixmap(pixmap)
                elif mode_flag ==1:
                    cascade_face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
                    haar_img=img
                    image_center = (width // 2, height // 2)
                    cv2.circle(haar_img, image_center, 5, (0, 255, 0), -1)
                    face_detections = cascade_face_detector.detectMultiScale(haar_img, scaleFactor=1.3, minNeighbors=4)

                    if len(face_detections) > 0 :
                        (x, y, w, h) = face_detections[0]
                        b, g, r = random.sample(range(256), 3)
                        cv2.rectangle(haar_img, (x, y), (x + w, y + h), (b, g, r), 2)
                        face_center = (x +(w // 2), y + (h // 2))
                        cv2.circle(haar_img, face_center, 5, (0, 0, 255), -1)
                        cv2.putText(haar_img, "Face", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (b, g, r), 2)
                    haar_img=cv2.cvtColor(haar_img, cv2.COLOR_BGR2RGB)
                    qHImg=QImage(haar_img.data,width,height,width*c,QImage.Format_RGB888)
                    pixmap=QPixmap.fromImage(qHImg)
                    label2.resize(width,height)
                    label2.setPixmap(pixmap)  
                
                elif mode_flag==2:
                    half_img = img[height // 2:, :]

                    lower_bound = np.array([0, 0, 0])
                    upper_bound = np.array([255, 255, 80])
                    mask = cv2.inRange(half_img, lower_bound, upper_bound)

                    M = cv2.moments(mask)
                    
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0
                    center_offset = width // 2 - cX

                    cv2.circle(half_img, (cX, cY), 10, (0, 255, 0), -1)
                    h,w,c1= half_img.shape
                    half_img=cv2.cvtColor(half_img, cv2.COLOR_BGR2RGB)
                    qImg_half=QImage(half_img.data,w,h,w*c1,QImage.Format_RGB888)
                    pixmap=QPixmap.fromImage(qImg_half)
                    label2.resize(w,h)
                    label2.setPixmap(pixmap)

                    if center_offset > 10:
                        print("오른쪽")
                        right()
                    elif center_offset < -10:
                        print("왼쪽")
                        left()
                    else:
                        print("직진")
                        forward()

        except:
            print("error")
            pass
                                      
#자동차 동작구현 함수
def forward():
    urlopen('http://'+ip+"/action?go=forward")
def stop():
    urlopen('http://'+ip+"/action?go=stop")
def backward():
    urlopen('http://'+ip+"/action?go=backward")
def right():
    urlopen('http://'+ip+"/action?go=right")
def left():
    urlopen('http://'+ip+"/action?go=left")    
def turn_right():
    urlopen('http://'+ip+"/action?go=turn_right")  
def turn_left():
    urlopen('http://'+ip+"/action?go=turn_left")  
def speed40():
    urlopen('http://'+ip+"/action?go=speed40")  
def speed50():
    urlopen('http://'+ip+"/action?go=speed50")    
def speed60():
    urlopen('http://'+ip+"/action?go=speed60")  
def speed80():
    urlopen('http://'+ip+"/action?go=speed80")  
def speed100():
    urlopen('http://'+ip+"/action?go=speed100") 

def haar_click():
    global mode_flag
    if mode_flag == 1:
        mode_flag= 0
    elif mode_flag!=1:
        mode_flag=1
def AI_click():
    global mode_flag
    if mode_flag == 2:
        mode_flag=0
    elif mode_flag!=2:
        mode_flag=2
def YOLO_click():
    print("YOLO")

class MyApp(QMainWindow):
    def __init__(self):
            super().__init__()
    def keyPressEvent(self, e):
        global mode_flag
        if e.key() == Qt.Key_W:
            forward()
        elif e.key() == Qt.Key_S:
            backward()
        elif e.key() == Qt.Key_A:
            left()
        elif e.key() == Qt.Key_D:
            right()
        elif e.key() == Qt.Key_Space:
            stop()
        elif e.key() == Qt.Key_H:
            if mode_flag == 1:
                mode_flag= 0
            elif mode_flag!=1:
                mode_flag=1
        elif e.key() == Qt.Key_T:
            if mode_flag == 2:
                mode_flag=0
            elif mode_flag!=2:
                mode_flag=2
    def keyReleaseEvent(self,e):
        stop()

class QPushButton(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setStyleSheet("color: white;"
                          "background-color: #2f4468")

app= QApplication(sys.argv)
win= MyApp()
widget = QWidget()
grid = QGridLayout()

widget.setLayout(grid)

label1=QLabel('DADUINO AI CAR')
title=QFont("Arial",20)
title.setBold(1)
label1.setFont(title)
label1.resize(200,200)
label1.setStyleSheet("border-style: solid;"
                      "border-width: 2px;"
                      "border-color: #FA8072;")
label2=QLabel()
label2.setStyleSheet("border-style: solid;"
                      "border-width: 2px;"
                      "border-color: #FA8072;")

btn_speed40=QPushButton('Speed40')
btn_speed50=QPushButton('Speed50')
btn_speed60=QPushButton('Speed60')
btn_speed80=QPushButton('Speed80')
btn_speed100=QPushButton('Speed100')

hbox1=QHBoxLayout()
hbox1.addWidget(btn_speed40)
hbox1.addWidget(btn_speed50)
hbox1.addWidget(btn_speed60)
hbox1.addWidget(btn_speed80)
hbox1.addWidget(btn_speed100)

btn_forward=QPushButton('forward')

btn_tleft=QPushButton('Trun Left')
btn_left=QPushButton('Left')
btn_stop=QPushButton('Stop')
btn_right=QPushButton('Right')
btn_tright=QPushButton('Turn Right')

hbox2=QHBoxLayout()
hbox2.addStretch(1)
hbox2.addWidget(btn_tleft)
hbox2.addWidget(btn_left)
hbox2.addWidget(btn_stop)
hbox2.addWidget(btn_right)
hbox2.addWidget(btn_tright)
hbox2.addStretch(1)

btn_backward=QPushButton('Backward')

hbox3=QHBoxLayout()
btn_haar=QPushButton('Haar')
btn_AI=QPushButton('AI Mode')
btn_YOLO=QPushButton('yolo')
hbox3.addWidget(btn_haar)
hbox3.addWidget(btn_AI)
hbox3.addWidget(btn_YOLO)

grid.addWidget(label1, 0, 0,alignment=Qt.AlignHCenter)

grid.addWidget(label2, 1, 0,alignment=Qt.AlignHCenter)

grid.addLayout(hbox1,2,0)

grid.addWidget(btn_forward, 3, 0,alignment=Qt.AlignHCenter)

grid.addLayout(hbox2,4,0)
grid.addWidget(btn_backward, 5, 0,alignment=Qt.AlignHCenter)

grid.addLayout(hbox3,6,0)

deamon_thread=threading.Thread(target=streaming_thread)
deamon_thread.daemon=True
deamon_thread.start()

win.setCentralWidget(widget)
win.setWindowTitle('AI CAR PRACTICE')
win.move(400, 200)
win.resize(800, 800)
win.show()

btn_forward.clicked.connect(forward)
btn_stop.clicked.connect(stop)
btn_backward.clicked.connect(backward)
btn_right.clicked.connect(right)
btn_left.clicked.connect(left)
btn_tright.clicked.connect(turn_right)
btn_tleft.clicked.connect(turn_left)
btn_speed40.clicked.connect(speed40)
btn_speed50.clicked.connect(speed50)
btn_speed60.clicked.connect(speed60)
btn_speed80.clicked.connect(speed80)
btn_speed100.clicked.connect(speed100)
btn_haar.clicked.connect(haar_click)
btn_AI.clicked.connect(AI_click)
btn_YOLO.clicked.connect(YOLO_click)

sys.exit(app.exec_())
        



