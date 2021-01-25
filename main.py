import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# getting all images name from images folder
path = 'images'
images = []
imgNames = []
# getting files from path as a list
mylist = os.listdir(path)

# importing images
for img in mylist:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    imgNames.append(os.path.splitext(img)[0])

print(f'imgNames: {imgNames}')

# encoding images
def encode_image(images):
    encoded_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_img = face_recognition.face_encodings(img)[0]
        encoded_images.append(encoded_img)
    return encoded_images


encoded_image_list = encode_image(images)
print('Encoding complete')


# initializing webcam from webcam
web_cam = cv2.VideoCapture(0)

def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not  in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S - %D')
            f.writelines(f'\n{name},{date_string}')



while True:
    success, img = web_cam.read()
    imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # get  web cam encoding
    facesCurFrame = face_recognition.face_locations(imgS)
    encoded_curFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Compare encoded images
    for encode_face, face_loc in zip(encoded_curFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encoded_image_list, encode_face)
        face_distance = face_recognition.face_distance(encoded_image_list, encode_face)
        match_index = np.argmin(face_distance)

        print(match_index)

        try:
            if matches[match_index]:
                name = imgNames[match_index].upper()
                print(name)

                # draw rectangle around the image
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (66, 133, 244), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (66,133,244), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        except Exception as e:
            print(e)
    if cv2.waitKey(1) == 27:
        break




    cv2.imshow('webcam', img)
    cv2.waitKey(1)