import cv2

video_c = cv2.VideoCapture('Dhaka City Gulshan 2 to 1 Video Road View Raid Vlogs.mp4')

#pre trained classifer
classifier_file_c = 'cars.xml'

# creat a classifir
c_tracker = cv2.CascadeClassifier(classifier_file_c)

while True: 
    # Read the current frame
    (read_successful, frame) = video_c.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars
    car = c_tracker.detectMultiScale(grayscaled_frame)
    print(car)

    # draw rectanglers around the cars
    for (x,y,w,h) in car:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)

    # DIsplay the image with the face spotted
    cv2.imshow('Clever Program Car Detector', frame)

    # Dont autoclose and wait till a key press
    cv2.waitKey(1)

print("ok")