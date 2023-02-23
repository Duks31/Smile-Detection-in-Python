### FOR IMAGES 

# # imports 
import cv2

# # load some pretrained data on smile frontals from opencv (harr-cascade algorithm)
trained_smile_data = cv2.CascadeClassifier("haarcascade_smile.xml")


# # choose an image to detect smile in 
img = cv2.imread('smilechild.jpg')


# # must convert to grey-scale 
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# # detect smile
smile_coordinates = trained_smile_data.detectMultiScale(grayscale_img)    


# # drawing rectangle around the smile 
for (x, y, w, h) in smile_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255,142), 4)
# # shape       img     img coordinates    color    thickness    


# # show the image 
cv2.imshow('My Smile Detector', img)


# this is to display the image until a key is pressed  
cv2.waitKey(0)


#### FOR WEBCAM 


import cv2

#pretrained algorithm 
classifier_file = 'haarcascade_smile.xml'

#create classifier 
smile_tracker = cv2.CascadeClassifier(classifier_file)

#capture video
webcam = cv2.VideoCapture(0)

while True:
    #Read  the current frame 
    (read_successful, frame) = webcam.read()

    if read_successful:
        #convert to black and white (grey scale)
        grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect smile 
    smile = smile_tracker.detectMultiScale(grey_scale)
    

    #Draw the rectangle around the smile
    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x,y),((x+w), (y+h)), (0, 255, 0), 3)

    #Show the image with the smile spotted
    cv2.imshow('My smile detection app', frame)

    #to avoid immediate close, close only on key press 
    cv2.waitKey(1)

# to avoid closure immediately 
cv2.waitKey(1)

