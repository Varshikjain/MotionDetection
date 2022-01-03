import cv2,time,pandas
from numpy import true_divide
from datetime import datetime
first_name=None
status_list=[None,None]
times=[]
#dataframe to store the time values during which object detection and movement appears
df=pandas.DataFrame(columns=["Start","End"])
#method to create video capture object.It triggers camera
video=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
        check,frame=video.read()
        #status of beginning is 0 when there was no object
        status=0
        #Converting frame color to gray scale
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #converting grey scale image to gaussian blur image
        gray=cv2.GaussianBlur(gray,(21,21),0)
        #stores first image/frame of video
        if first_name is None:
            first_name=gray
            continue
        #calculate difference bw first and other frames
        delta_frame=cv2.absdiff(first_name,gray)
        #providing threshold value that with value less than 30 will be black and white if greater than it
        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        #adding borders to object that appears
        (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        #removes unwanted small object, will keep those having area greater than 1000px
        for contour in cnts:
           if cv2.contourArea(contour) < 10000:
             continue
           #changing status when object is being detected 
           status=1
           #creating rectangular box
           (x, y, w, h)=cv2.boundingRect(contour)
           cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        status_list.append(status)#list of status for every frame

        status_list=status_list[-2:]

#record datetime in list when object occurs
        if status_list[-1]==1 and status_list[-2]==0:
           times.append(datetime.now())
        if status_list[-1]==0 and status_list[-2]==1:
           times.append(datetime.now())

        #showing 4 different frames 
        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Delta Frame",delta_frame)
        cv2.imshow("Threshold Frame",thresh_frame)
        cv2.imshow("Color Frame",frame)
        #changes frame after 1 millisec
        key=cv2.waitKey(1)
        #closes the window when pressed
        if key==ord('q'):
          if status==1:
            times.append(datetime.now())
          break

print(status_list)
print(times)

for i in range(0,len(times),2):
    #storing time values in dataframe
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")#writing to csv file

video.release()
cv2.destroyAllWindows


