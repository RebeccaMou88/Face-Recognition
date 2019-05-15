import numpy as np
import cv2
import time
import os

label_text = []
person_names = [""]


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convertToGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#draw rectangle on image
def draw_rectangle(img, rect):
    #print(len(img))
    #for rect in rects:
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#drawing text from x and y cordinate
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 255, 0), 2)

#face detecting 
def face_detect(img, scaleFactor=1.3, minNeighbors=5, returnGray=True):
    img_copy = img.copy()
    gray = convertToGray(img_copy)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors);

    
    if (len(faces) == 0):
        return None, None

    #print(faces[0])
    #print(gray)
    
    result = []

    for (x, y, w, h) in faces:
        result.append((x, y, w, h))

    if returnGray:
         return gray, result
    else:
        return img, result
    
    #(x, y, w, h) = faces[0]
 
    #return gray[y:y+w, x:x+h], faces[0]

#reading training data
def reading_training_data(training_folder_path):

    dirs = os.listdir(training_folder_path)    

    faces = []
    labels = []

    
    for dir_name in dirs:
        person_names.append(dir_name.split("-")[1])

        subject_dir_path = training_folder_path +"/"+dir_name

        images_names = os.listdir(subject_dir_path)


        
        for image_name in images_names:
            if image_name.startswith("."):
                continue

            #image for each images
            image_path = subject_dir_path+"/"+image_name
            
            label = int(dir_name.split("-")[0].replace("person",""))
            
            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(1)

            #face detect
            face, rect = face_detect(image)

            if face is not None:
                #print(len(face))
                #print(rect)
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
        
    
print("Reading data...")
faces,labels = reading_training_data('training-data')
print("Data read completed")

#print(faces)
#print(labels)
person_names.append("Unknown")
print(person_names)

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#face_recognizer.save("trainer.yml")

def predicts(test_img):
    img = test_img.copy()

    face, rect = face_detect(img)

    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text.append(person_names[label])

        for rect in rect:
            draw_rectangle(img, rect)
            draw_text(img, person_names[label], rect[0], rect[1]-5)

    return img

def predictVideo(video=0):

    cap = cv2.VideoCapture(video)
    
    while(True):
        ret, img = cap.read()

        #print(img)
        if img is not None:
            face, rect = face_detect(img)

            if face is not None:
                label, confidence = face_recognizer.predict(face)
                label_text.append(person_names[label])
                for rect in rect:
                    draw_rectangle(img, rect)
                    #draw_text(img, person_names[label], rect[0], rect[1]-5)

                cv2.imshow("Video", cv2.resize(img, (700,500)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()

    return 0


print("Predicting images...")


#test_img1 = cv2.imread("20170815_154737.jpg")
#print(test_img1)
#predicted_img1 = predicts(test_img1)

predicted_img1 = predictVideo(0)


print(label_text)

if len(label_text)>0:
    cv2.imshow(label_text[0], predicted_img1)
else:
    cv2.imshow(person_names[-1], predicted_img1)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()


