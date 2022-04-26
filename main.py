import math
import socket
import cv2
import numpy as np
import threading
from lib import *
global sendBack_angle, sendBack_Speed, current_speed, current_angle
global imaged, label, label_lag, label_max,to1,to2
imaged=None
label=None
global label_array
label_max=[]
label_class=[]
thre=0.1
label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
sendBack_angle = 0
sendBack_Speed = 0
current_speed = 0
current_angle = 0
to1=0
to2=0
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 54321
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

def nhan_dang():
    global imaged ,label,label_lag, label_array,label_max
    while(True):
        if imaged is not None:
            #print('hehe')
            i = 0
            whT = 480
            confThreshold = 0.3
            nmsThreshold = 0.3
            classesFile = 'yolo.names'
            classNames = []
            with open(classesFile, 'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')
            modelConfiguration = 'yolov4-tiny-custom.cfg'
            modelWeights = 'yolov4-tiny-custom_last2.weights'
            net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            layerNames = net.getLayerNames()
            outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
            while(True):
                blob = cv2.dnn.blobFromImage(imaged, 1/255, (whT, whT), [0, 0, 0], 1, crop = False)
                net.setInput(blob)
                outputs = net.forward(outputNames)
                imaged,indices,classIds,confs=car.findObject(outputs, imaged)
                if len(indices)==0:
                    label=None
                if len(indices) !=0:
                       for i in indices:
                            i = i[0]
                            label=classIds[i]
                            #print(label)
                            if label is not None:
                                label_array[label]=label_array[label]+confs[i]
                                label_max.append([label,confs[i]])
                                #print(confs[i],(label_array[label]),label,label_array)

def Control(angle, speed):
    global sendBack_angle, sendBack_Speed
    sendBack_angle = angle
    sendBack_Speed = speed
    
def tinhdientich(img,image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    dientich=0
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        if w*h < 15000:
            cv2.drawContours(mask, [i], -1, 0, -1)
        else:
            dientich=+dientich+w*h
    return dientich,mask
    
x1_right =None
x1_left = None
lines=[]
mode=0
if __name__ == "__main__":
    t1 = threading.Thread(target=nhan_dang)
    t1.start()
    try:
        while True:
            """
            - Chương trình đưa cho bạn 1 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [-150, 150]
                        NOTE: (âm là lùi, dương là tiến)
            """

            message_getState = bytes("0", "utf-8")
            s.sendall(message_getState)
            state_date = s.recv(100)

            try:
                current_speed, current_angle = state_date.decode(
                    "utf-8"
                    ).split(' ')
            except Exception as er:
                print(er)
                pass

            message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
            s.sendall(message)
            data = s.recv(100000)
            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                        ), -1
                    )
                #print(t1)
                #Control(0, 0)
                car = auto_car(image)
                image = np.asarray(image)
                i=image[50:250,250:550]
                #i=car.bird_view(i)
                imaged=cv2.resize(i,[480,320])
                bird=car.bird_view1(image)
                bird=bird[0:150,10:290]
                bird=cv2.resize(bird,[300,250])
                image = car.take_blue(bird, bird)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#gray
                ret,gray = cv2.threshold(image,100,255,cv2.THRESH_BINARY)#nhi phan
                img = np.copy(gray)
                cv2.imshow("be4",img)
                dientich,mask=tinhdientich(img,image)#lọc thành phần nhỏ
                gray = cv2.bitwise_and(gray, gray, mask=mask)
                image = cv2.bitwise_and(image, image, mask=mask)
                cv2.imshow("gray",image)
                #cv2.imshow("mask", mask)
                lenght=[]
                find_max3=[]
                find_min3=[]
                avr=[]
                cv2.waitKey(1)
                arr=np.sum(image,axis=0)#sum dọc ảnh
                for i in range(len(arr)-2):
                    avr.append(arr[i]-arr[i+2])# dạo hàm 
                avr=np.sum(avr)/len(avr)
                for i in range(len(arr)-2):
                    if abs(arr[i]-arr[i+2])>(avr):
                        lenght.append(i+1)
                        find_max3.append([i+1,arr[i+1]])
                #print(lenght)
                if len(find_max3)>15:
                    find_max3=sorted(find_max3, key=lambda x:x[1])
                    find_max3=find_max3[-int(len(find_max3)/2):]
                    find_max3=sorted(find_max3, key=lambda x:x[0])
                    find_min3=find_max3[:10]
                    find_max3=find_max3[-10:]
                    canny_image = cv2.Canny(gray, 100, 255)
                    #cv2.imshow("canny",canny_image)
                    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 30,None,80,10)
                    right=[];
                    left=[];
                    #print(N>0)
                    if lines is not None:
                        if (len(lines)>0):
                            N=len(lines)
                            for i in range(N):
                                x1 = lines[i][0][0]
                                y1 = lines[i][0][1]
                                x2 = lines[i][0][2]
                                y2 = lines[i][0][3]
                                cv2.line(bird, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                sqrt=math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))
                                if y1 > y2:
                                    a = x2 - x1
                                    b = y2 - y1
                                else:
                                    a = x1 - x2
                                    b = y1 - y2
                                x = (a*(250-y1)+b*x1)/(b+0.0000000000000001)
                                #print((abs(x-lenght[1])))
                                #print(x)
                                if len(find_max3)>0:
                                    for j in range(9):
                                        if((abs(x-find_max3[j][0]))<80)and((x>100)and(x<280)):
                                            #print(math.acos((x1-x2)/sqrt)*180/np.pi)
                                            right.append([x1,y1,x2,y2,x,180*math.acos((a)/sqrt)/np.pi])
                                            #print('right')
                                if len(find_min3)>0:
                                    for j in range(9):
                                        if((abs(x-find_min3[j][0]))<80)and((x>20)and(x<200)):
                                            #print(math.acos((x1-x2)/sqrt)*180/np.pi)
                                            left.append([x1,y1,x2,y2,x,180*math.acos((a)/sqrt)/np.pi])
                                            #print('left')
                            lenn_right=0.1
                            lenn_left=0.1
                            lenn_right=(len(right)+0.000000000001)
                            lenn_left=(len(left)+0.000000000001)
                            #print(lenn)
                            if lenn_right>=1:
                                right=(np.sum(right,axis=0))
                                right=right/lenn_right
                                #print(lenn)
                                if right[4]>500:
                                    #print('1')
                                    x1_right = right[0]
                                    y1_right = right[1]
                                    x2_right = right[2]
                                    y2_right = right[3]
                                    cv2.line(bird, (int(x1_right), int(y1_right)), (int(x2_right), int(y2_right)), (255, 0, 0), 2)
                                else:
                                    #print('2')
                                    theta = right[5]
                                    x1_right=right[4]
                                    y1_right=250
                                    y2_right=y1_right-100*math.sin(float(np.pi*theta/180))
                                    x2_right=x1_right+100*math.cos(float(np.pi*theta/180))
                                    cv2.line(bird, (int(x1_right), int(y1_right)), (int(x2_right), int(y2_right)), (255, 0, 0), 2)
                                #print(x1, y1, x2, y2)
                                #gocc, vantocc = car.PDfuzzy((x1+x2)/2-200)
                                #Control(gocc, vantocc)
                            if lenn_left>=1:
                                left=(np.sum(left,axis=0))
                                left=left/lenn_left
                                #print(lenn_left)
                                if left[4]<-200:
                                    x1_lefr = left[0]
                                    y1_left = left[1]
                                    x2_left = left[2]
                                    y2_left = left[3]
                                    cv2.line(bird, (int(x1_left), int(y1_left)), (int(x2_left), int(y2_left)), (0, 0,255), 2)
                                else:
                                    theta = left[5]
                                    #print(theta)
                                    x1_left=left[4]
                                    y1_left=250
                                    y2_left=y1_left-100*math.sin(float(np.pi*theta/180))
                                    x2_left=x1_left+100*math.cos(float(np.pi*theta/180))
                                    cv2.line(bird, (int(x1_left), int(y1_left)), (int(x2_left), int(y2_left)), (0, 0, 255), 2)
                                #print(x1, y1, x2, y2)
                                #gocc, vantocc = car.PDfuzzy((x1+x2)/2-200)
                                #Control(gocc, vantocc)
                            cv2.imshow('line', bird)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #print(current_speed, current_angle)
                #print(image.shape)
                # your process here
                #cv2.imshow("IMG", imaged)
                #Control(angle, speed)
##                blob = cv2.dnn.blobFromImage(imaged, 1/255, (whT, whT), [0, 0, 0], 1, crop = False)
##                net.setInput(blob)
##                outputs = net.forward(outputNames)
##                imaged,indices,classIds=car.findObject(outputs, imaged)
                #gocc,vantocc=car.control_car(indices,classIds,x1);
                x,y=car.hoiqui(gray,image)
                if len(label_max)>0:
                    label1=sorted(label_max, key=lambda x:x[1])
                    label_moni=label1[len(label1)-1][0]
                else:
                    label_moni=None
                print(label_moni,label_max, (time.time()-to1)%100, label)
                print(mode)
                #print(np.argmax(np.array(label_array)),label )
                result= np.argmax(label_array)
                if (dientich>69000)and((label_moni==7))and(mode!=2)and(mode!=3)and(mode!=4):
                    mode=1
                if (dientich>69000)and(label_moni==8)and(mode!=1)and(mode!=3)and(mode!=4):
                    mode=2
                if ((label_moni==10)or()and(mode!=1)and(mode!=2)and(mode!=4)):
                    mode=3
                if ((label_moni==9)or(label_moni==2)or(label_moni==4))and(mode!=1)and(mode!=2)and(mode!=3):
                    mode=4
                if mode==0:
                    gocc, vantocc = car.PDfuzzy((x)-150)
                    Control(gocc, vantocc)
                    to2=time.time()
                    if label is None:
                        #label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                        label_class=[]
                        #print((time.time()-t)%100)
                        if float((time.time()-to1))%100>3:
                            label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                            label_max=[]
                    if (label is not None)or(label_moni==1):
                        label_class.append([label])
                        to1=time.time()
                        if len(label_class)>2:
                            for t in range(5):
                                Control(gocc,-3)
                                time.sleep(0.001)
                            #label_array=[0,0,0,0,0,0,0,0,0,0,thre]
                            #label=None
                            #label_class=[]
                if mode==1:
                    gocc, vantocc = car.PDfuzzy((x)-220)
                    Control(gocc, vantocc)
                    if float((time.time()-to2))%100>0.1:
                        if (x1_left is not None)and(dientich<67000):
                            mode=0
                            label=None
                            label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                            label_max=[]
                if mode==2:
                    gocc, vantocc = car.PDfuzzy((x)-80)
                    Control(gocc, vantocc)
                    if float((time.time()-to2)%100)>0.1:
                        if (x1_right is not None)and(dientich<67000):
                            mode=0
                            label=None
                            label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                            label_max=[]
                if mode==3:
                    gocc, vantocc = car.PDfuzzy(x-200)
                    Control(gocc, vantocc)
                    if float((time.time()-to2)%100)>4:
                        if (x1_right is not None)and(dientich<67000):
                            mode=0
                            label=None
                            label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                            label_max=[]
                if mode==4:
                    gocc, vantocc = car.PDfuzzy(x-100)
                    Control(gocc, vantocc)
                    if float((time.time()-to2)%100)>4:
                        if (x1_right is not None)and(dientich<67000):
                            mode=0
                            label=None
                            label_array=[0,0,0,0,0,0,0,0,0,0,0,thre]
                            label_max=[]
                cv2.waitKey(1)

            except Exception as er:
                print(er)

    finally:
        print('closing socket')
        t1.join()
        s.close()
