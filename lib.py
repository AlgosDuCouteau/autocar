import numpy as np
import cv2
import time
import math
error_arr = np.zeros(5)
error_fuzzy = np.zeros(5)
I=np.zeros(5)
dt1 = time.time()
dt2 = time.time()
dt3 = time.time()
clock = 110
set_time = 0
i = 0
whT = 320
confThreshold = 0.3
nmsThreshold = 0.3
classesFile = 'yolo.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
modelConfiguration = 'yolov4-tiny-custom.cfg'
modelWeights = 'yolov4-tiny-custom_final.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerNames = net.getLayerNames()
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
class auto_car():
        def __init__(self,image):
                self.img=image

        def mean_ignore_zero(chanel):
                return chanel[np.nonzero(chanel)].mean()

        def make_coordinates(image, line_parameters):

                a, b, c = line_parameters
                t = image.shape[0]
                y1 = int(2 * t / 3)
                y2 = int(t / 4)
                y3 = int(t / 2)
                x1 = int((-b * y1 - c) / a)
                x2 = int((-b * y2 - c) / a)
                x3 = int((-b * y3 - c) / a)
                # print(x1,x2)
                return np.array([x1, y1, x2, y2, x3, y3])
        def can(self,image):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                B=np.ones_like(gray)
                kernel1 =np.ones((5,5),np.uint8);
                kernel2 =np.ones((5,5),np.uint8);
                B = cv2.erode(B,kernel1,iterations = 1)
                B = cv2.dilate(B,kernel2,iterations = 1)
                gray=gray*B
                #gray=gray*(gray>5)
                #gray=cv2.equalizeHist(gray)
                #cv2.imshow('hihi',B*255)
                blur = cv2.medianBlur(gray, 7)
                #blur = cv2.GaussianBlur(gray, (7, 7), 0)
                #if abs(r)<0.2:
                #   x_av=150
                return gray
    
        def bird_view(self,image):
               #image=self.img
                x = image.shape[0]
                y = image.shape[1]
                
                x0, y0 = x    , 0
                x1, y1 = x-150, 0
                x2, y2 = x-150, y-70
                x3, y3 = x    , y+70
                #cv2.line(image, (y0,x0), (y1,x1), (0, 0, 255), 1)
                #cv2.line(image, (y3,x3), (y2,x2), (0, 0, 255), 1)

                src = np.float32([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
                dst = np.float32([[10, 350], [10, 100], [400, 100], [400, 350]])
                m = cv2.getPerspectiveTransform(src, dst)
                birdview = cv2.warpPerspective(image, m, (300, 250))

                #cv2.imshow("src",image)
                #cv2.imshow("bird view",birdview)                return birdview
                return birdview
        
        def bird_view1(self,image):
                #image=self.img
                x = image.shape[0]
                y = image.shape[1]
                
                x0, y0 = x-100, 0
                x1, y1 = x-150, 0
                x2, y2 = x-150, y
                x3, y3 = x-100, y
                #cv2.line(image, (y0,x0), (y1,x1), (0, 0, 255), 1)
                #cv2.line(image, (y3,x3), (y2,x2), (0, 0, 255), 1)

                src = np.float32([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
                dst = np.float32([[10, 200], [10, 50], [290, 50], [290, 200]])
                m = cv2.getPerspectiveTransform(src, dst)
                birdview = cv2.warpPerspective(image, m, (300, 250))

                #cv2.imshow("src",image)
                #cv2.imshow("bird view",birdview)
                return birdview

        def ROI(self,image):
                #image=self.img
                height = image.shape[0]
                shape = np.array([[25,125], [275, 125], [300, 300],[0,300]])
                mask = np.zeros_like(image)
                if len(image.shape) > 2:
                    channel_count = image.shape[1]
                    ignore_mask_color = (255,) * channel_count
                else:
                    ignore_mask_color = 255
                cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
                masked_image = cv2.bitwise_and(image, mask)
                return masked_image

        def PID(self,error, p=0.1, i=0.08, d=0.9):
                global dt1
                global error_arr
                error_arr[1:] = error_arr[0:-1]
                error_arr[0] = error
                P = error * p
                delta_t = time.time() - dt1
                #print(delta_t)
                dt1 = time.time()
                D = (error - error_arr[1]) / delta_t * d
                #I = np.sum(error_arr) * delta_t * i
                #print (error)
                if error>30:
                    A=error-25;
                    I = np.sum(error_arr) * delta_t * i
                elif error<-30:
                    A=error+25;
                    I = np.sum(error_arr) * delta_t * i
                else:
                    A=0;
                    I=0
                angle = P  + D+I
                return -int(angle)


        def display_lines(self,image,lines_r,lines_l):
                #image=self.img
                line_image = np.zeros_like(image)
                setpoint=150
                if lines_r is not None:
                    x1_r, y1_r, x2_r, y2_r, x3_r,y3_r = lines_r.reshape(6)
                    cv2.line(line_image, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 10, 8)
                    xa_r = int((x1_r*y1_r + x2_r*y2_r+x3_r*y3_r) / (y3_r+y1_r+y2_r))  # red 110 160
                    ya_r = int((y1_r + y2_r) / 2)
                    #cv2.circle(line_image, (xa_r - 100, 280), 5, (0, 255, 0), -1)
                    #cv2.circle(line_image, (150, 280), 5, (0, 0, 255), -1)
                if lines_l is not None:
                    x1_l, y1_l, x2_l, y2_l,x3_l,y3_l = lines_l.reshape(6)
                    cv2.line(line_image, (x1_l, y1_l), (x2_l, y2_l), (0, 255, 0), 10, 8)
                    xa_l = int((x1_l*y1_l+ x2_l*y2_l+x3_l*y3_l) / (y3_l+y1_l+y2_l))  # red 110 160
                    ya_l = int((y1_l + y2_l) / 2)
                #x_d = float(xa_l - 100)  # green 150
                a=60
                if (lines_r is None)and (lines_l is not None):
                    setpoint=xa_l+a
                if (lines_r is not None)and (lines_l is not None):
                    if (x2_r-x2_l>70)and(x2_r-x2_l<100):
                        setpoint=int(((xa_r-75)+(xa_l+75))/2)
                    elif(x2_r-x2_l<150)and(x2_r-x2_l>=100):
                        setpoint=xa_l-a
                    else:
                        setpoint=xa_r-a
                if (lines_r is not None)and (lines_l is None):
                    setpoint=xa_r-a
                if (lines_r is None)and (lines_l is None):
                    setpoint=150
                goc,vantoc=self.PDfuzzy(150-setpoint)
                if (lines_r is None)and (lines_l is None):
                    goc=0
                    vantoc=20
                #cv2.circle(line_image, (setpoint, 280), 5, (0, 255, 0), -1)
                #cv2.imshow('line',line_image)
                return line_image, goc,vantoc


        def average_slope_intercept(self,image, lines):
                #image=self.img
                right_fit = []
                left_fit=[]
                right_line=None
                left_line=None
                if lines is None:
                    return None, None
                for line in lines:
                    rho,theta = line[0]
                    a=float(np.cos(theta))
                    b=float(np.sin(theta))
                    x1=a*rho;
                    y1=b*rho;
                    c=float(-rho);#n(a,b) ax+by+c=0
                    theta=theta/np.pi*180
                    if  (theta >150):
                        #print(a)
                        right_fit.append((a, b,c))
                    if  (theta <30):
                        #print(a)
                        left_fit.append((a, b,c))
                # sap xep right_fit theo chieu tang dan cua intercept
                leng = len(right_fit)
                right_fit = np.array(sorted(right_fit, key=lambda a_entry: a_entry[0]))
                right_fit = right_fit[::-1]
                right_fit_focus = right_fit
                if leng > 2:
                    right_fit_focus = right_fit[:1]
                if len(right_fit_focus)>0:
                    right_fit_average = np.array(np.average(right_fit_focus,axis=0))
                    right_line = self.make_coordinates(image, right_fit_average)
                # sap xep right_fit theo chieu tang dan cua intercept
                leng = len(left_fit)
                left_fit = np.array(sorted(left_fit, key=lambda a_entry: a_entry[0]))
                left_fit = left_fit[::-1]
                left_fit_focus = left_fit
                if leng > 2:
                    left_fit_focus = left_fit[:1]
                if len(left_fit_focus)>0:
                    left_fit_average = np.array(np.average(left_fit_focus,axis=0))
                    left_line =self.make_coordinates(image, left_fit_average)
                return right_line,len
        def hoiqui(self,image,anh_gray):
                #gray=self.img
                gray=image
                ANH=anh_gray
                I=(gray>0)*1
                a=I.shape[1]
                b=I.shape[0]
                x=np.arange(1,a+1)
                x=np.array(x)
                y=np.arange(1,b+1)
                y=np.array(y)
                t_x=[]
                T=np.sum(I,axis=1)
                T1=T*(T>0)+1*(T==0)
                G=np.sum(x*I,axis=1)
                T_x=np.dtype(np.int64)
                t_x=G/T1
                t_x=t_x*(T>0)+b/2*(T==0)
                t_x=np.array(t_x)
                x=t_x
                n=b
                x_av=(1/n)*sum(x)
                y_av=(1/n)*sum(y)
                xy_av=(1/n)*sum(x*y)
                xx_av=(1/n)*sum(x*x)
                yy_av=(1/n)*sum(y*y)
                sx=np.sqrt(xx_av-x_av*x_av)
                sy=np.sqrt(yy_av-y_av*y_av)
                r=(xy_av-x_av*y_av)/(sx*sy+0.00000000000001)#x=by+a
                #print('do tuong quan %d',r)
                b1=(xy_av-x_av*y_av)/(sy*sy+0.0000000000001)
                a1=x_av-b1*y_av
                #print(a,b)
                y1=0
                x1=int(a1)
                y2=b
                x2=int(y2*b1+a1)
                #print(x1,x2)
                y0=b/2
                x0=int(y0*b1+a1)
                cv2.line(ANH, (x1, y1), (x2, y2), (255, 0, 0), 10, 8)
                cv2.imshow('line3',ANH)
                return x0,y0

    
        def enhance_contrast(self,image):
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                gamma = math.pow(math.log(self.mean_ignore_zero(lab[:,:,0]))/math.log(80),5)
                lookUpTable = np.empty((1,256), np.uint8)
                for i in range(256):
                    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
                    res = cv2.LUT(image, lookUpTable)
                    show = np.hstack((image,res))
                    #cv2.imshow("gamma",cv2.cvtColor(show,cv2.COLOR_RGB2BGR))
                    #check = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)
                    #print('{} : {}'.format(mean_ignore_zero(lab[:,:,0]),mean_ignore_zero(check[:,:,0])))
                return res
        def take_blue(self,image,fram):
                hsv_frame=hsv_frame=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                low_blue=np.array([50,0,2])#94 80 2
                high_blue=np.array([126,255,255])#126 255 255
                mask_blue=cv2.inRange(hsv_frame,low_blue,high_blue)
                
                blue=cv2.bitwise_and(fram,fram,mask=mask_blue)
                return blue
        def fuzzy(self,x,khoang):
                try:
                        a=khoang/2;
                        NB1=float(1*(x<=-2*a)+(-1/a*(x+a))*(-2*a<x<-a))
                        NS1=float((1/a*(x+2*a))*(-2*a<x<=-a)+(-1/a*(x))*(-a<x<=0))
                        ZE1=float((1/a*(x+a))*(-a<x<=0)+(-1/a*(x-a))*(0<x<=a))
                        PS1=float((1/a*(x))*(0<x<=a)+(-1/a*(x-2*a))*(a<x<=2*a))
                        PB1=float(1*(x>2*a)+(1/a*(x-a))*(a<x<=2*a))
                        return np.array([NB1,NS1,ZE1,PS1,PB1])
                except:
                        return np.array([0,0,0,0,0])
        def PDfuzzy(self,error):
                global error_fuzzy
                global I
                global dt2
                error_fuzzy[1:] =error_fuzzy[0:-1]
                error_fuzzy[0] = error
                e=error
                delta_t=time.time()-dt2
                dt2=time.time()
                #I[1:] =I[0:-1]
                #I[0] = I*delta_t
                I_p=sum(I)
                e_dot=(error-error_fuzzy[1])/delta_t
                #print(e_dot)
                e_fuzzy=self.fuzzy(e,50)
                e_fuzzy=e_fuzzy.reshape(-1,1)
                edot_fuzzy=self.fuzzy(e_dot,100)
                NB=-25
                NS=-8
                ZE=0
                PS=8
                PB=25
                x=e_fuzzy*edot_fuzzy
                #print(x)
                R=np.array([[NB,NB,NS,NS,ZE],
                            [NB,NS,NS,ZE,PS],
                            [NS,NS,ZE,PS,PS],
                            [NS,ZE,PS,PS,PB],
                            [ZE,PS,PS,PB,PB]])#pd
                e_v=self.fuzzy(e,50)
                e_v=e_v.reshape(-1,1)
                edot_v=self.fuzzy(e_dot,150)
                F=60
                L=40
                Z=0
                v=e_v*edot_v
                R1=np.array([[Z,Z,L,L,F],
                             [Z,L,L,F,L],
                             [L,L,F,L,L],
                             [L,F,L,L,Z],
                             [F,L,L,Z,Z]])
                y1=sum(sum(x*R))/sum(sum(x))#+I_p
                #print(sum(sum(v)))
                y2=sum(sum(v*R1))/sum(sum(v))
                return np.array([y1,y2])
        

        def control_car(self,indices,classIds,x1):
                global clock
                global set_time
                global dt3
                if len(indices)!=0:
                        print(indices)
                        gocc,vantocc=self.PDfuzzy(170-x1)
                        for i in indices:
                        #print(classIds[i])
                                i=np.array(i).reshape(1);
                                if classIds[i[0]]==6:
                                        gocc,vantocc=self.PDfuzzy(170-x1)
                                        dt3=time.time();
                                        print(clock);
                                        print(dt3%100)
                                        clock=170;
                                        set_time=5;
                                        return gocc,vantocc
                else:
                        if (time.time()-dt3)%100<set_time:
                                gocc, vantocc = self.PDfuzzy(clock - x1)
                        else:
                                clock=110;
                                gocc, vantocc = self.PDfuzzy(clock - x1)
                print(clock);
                print(dt3%100)
                return gocc, vantocc
        
        def findObject(self,outputs,img):
                hT, wT, cT = img.shape
                bbox = []
                classIds = []
                confs = []
                for output in outputs:
                        for det in output:
                                scores = det[5:]
                                classId = np.argmax(scores)
                                confidence = scores[classId]
                                if confidence > confThreshold:
                                        w, h = int(det[2]*wT), int(det[3]*hT)
                                        x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                                        bbox.append([x, y, w, h])
                                        classIds.append(classId)
                                        confs.append(float(confidence))
                                        
                indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
                #print(indices)
                for i in indices:
                        i = i[0]
                        box = bbox[i]
                        x, y, w, h = box[0], box[1], box[2], box[3]
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
                        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.imshow('train',img)
                cv2.waitKey(1)

                return img,indices,classIds,confs
                
