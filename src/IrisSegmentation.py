import cv2
import math
import numpy as np
import ExtractSector

# Load Haar Cascades to identify faces and eyes
face_cascade = cv2.CascadeClassifier('CascadeClassifiers/haarcascade_face.xml')
eye_cascade = cv2.CascadeClassifier('CascadeClassifiers/haarcascade_eye.xml')

# Obtain the livestream using device 0 (webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

while True:
    # Extract frames from the livestream. These are the images we're going to analyze.
    ret, img = cap.read()
    # If unable to gain access to the webcam then the program will stop running
    if ret is False:
        print("Error gaining access to the webcam")
        break

    # Get the grayscale versions of the images to improve results
    gray_roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Delete some of the noise from the image to improve its quality for the elaboration phase
    roi_gray_blur = cv2.medianBlur(gray_roi, 5)

    # Detect faces and eyes section:
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        # Draw rectangle around the detected faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # Add a tag above the face rectangle
        cv2.putText(img, "Face", (x-5, y-5), cv2.FONT_ITALIC, 0.4, (255, 255, 255))

        # Compute y coordinates to define the area containing the eyes section
        section1 = math.ceil(y+h/1.75)
        section2 = math.ceil(y+h/4.5)

        # Compute x coordinates to define the area containing the eyes section
        roi_color_eyes = img[section2:section1, x:x+w]
        roi_gray_eyes = roi_gray_blur[section2:section1, x:x+w]

        # Detect eyes
        height, width, _ = roi_color_eyes.shape
        eyes = eye_cascade.detectMultiScale(roi_color_eyes, 1.2, 10)
        for(ex, ey, eh, ew) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(roi_color_eyes, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 1)
            # Add tags above the eyes rectangles
            if ex+ew < width//2:
                cv2.putText(roi_color_eyes, "Right eye", (ex-3, ey-3), cv2.FONT_ITALIC, 0.4, (255, 255, 255))
            else:
                cv2.putText(roi_color_eyes, "Left eye", (ex - 3, ey - 3), cv2.FONT_ITALIC, 0.4, (255, 255, 255))
    # Notice: the eyes scan will be made only inside the eyes section to optimize the research and to save resources

    # Add text for GUI commands and show the livestream frame by frame
    cv2.putText(img, "Spacebar = Iris Segmentation  ESC = Exit", (0, 440), cv2.QT_FONT_NORMAL, 0.8, (255, 255, 255))
    cv2.imshow('img', img)

    # Wait user's inputs - based on the inputs some actions will be taken
    k = cv2.waitKey(30) & 0xff

    # ESC Button - Close the program and livestream
    if k == 27:
        print("ESC Button pressed - The program will stop")
        break

    # Spacebar Button - Compute eyes segmentations
    if k == 32:
        try:
            # Pupil and iris detection
            circles = cv2.HoughCircles(roi_gray_eyes, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 64, param1=150, param2=15,
                                       minRadius=5, maxRadius=15)
            if circles is not None:
                circles = np.uint16(np.around(circles))
            # We will use a j flag to distinguish which eye is being processed (0 = right, 1 = left)
            j = 0
            for i in circles[0, :]:
                if j == 0:
                    right_eye = roi_color_eyes[i[1] - (i[2]-i[2]//3):i[1] + (i[2]-i[2]//3),
                         i[0] - (i[2]-i[2]//3):i[0] + (i[2]-i[2]//3)]
                    cv2.imwrite("ProcessingImages/RightSeg.png", right_eye)
                    j = 1
                else:
                    left_eye = roi_color_eyes[i[1] - (i[2]-i[2]//3):i[1] + (i[2]-i[2]//3),
                         i[0] - (i[2]-i[2]//3):i[0] + (i[2]-i[2]//3)]
                    cv2.imwrite("ProcessingImages/LeftSeg.png", left_eye)

            # Save all images gained from the processing phases
            cv2.imwrite("ProcessingImages/EyeSection.png", roi_color_eyes)
            cv2.imwrite("ProcessingImages/LeftEye.png", roi_color_eyes[0:height, width//2:width])
            cv2.imwrite("ProcessingImages/RightEye.png", roi_color_eyes[0:height, 0:width // 2])

            # Compute segmentation
            ExtractSector.segmentation()
            break

        except (TypeError, NameError):
            print("Error while processing. Try again")

cap.release()
cv2.destroyAllWindows()