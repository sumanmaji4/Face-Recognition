import numpy as np
import os
import cv2
import pickle
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList = [entry.split(',')[1] for entry in myDataList]  #Extract existing names

        #Updating serial number for the new entry
        last_serial = 0  #Initialize with 0
        if myDataList:
            last_entry = myDataList[-1].split(',')
            if last_entry[0].isdigit():  # Check if it's a valid serial number
                last_serial = int(last_entry[0])  # Get the last serial number

        serial_no = last_serial + 1  # Increment for the new entry

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{serial_no},{name},{dtString}')
            nameList.append(name)



video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img) #Output of the name of the known Faces
        nameAtt = str(output[0])
        markAttendance(nameAtt)
        cv2.putText(frame, str(output[0]), (x, y+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    #Quit from the webcam: PRESS x
    if k == ord('x'):
        break
video.release()
cv2.destroyAllWindows()