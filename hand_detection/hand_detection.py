
import cv2 as cv  # Import the OpenCV library
import mediapipe as mp  # Import the MediaPipe library 
import time


cap = cv.VideoCapture(0)  # Open the default camera


mpHands = mp.solutions.hands  # Create a hands object
hands = mpHands.Hands()  # Create a hands object
mpDraws = mp.solutions.drawing_utils  # Create a drawing object



pTime = 0


while True:
    success, img = cap.read()  # Capture a frame from the camera
    

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert the BGR image to RGB
    results = hands.process(imgRGB)  # Process the image
    print(results.multi_hand_landmarks)  # Print the landmarks of the hand

    if results.multi_hand_landmarks:
        for handlems in results.multi_hand_landmarks:

            for id, lm in enumerate(handlems.landmark):
                height, weight, channnels = img.shape
                cx, cy = int(lm.x*weight), int(lm.y*height) ## Covert decimals into pixels
                print(id, cx, cy)

                if id == 0:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            mpDraws.draw_landmarks(img, handlems, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    ### To put text on the image
    ## image, textToPut, Position, font_style, font_size, color_of_text, thickness_of_text
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv.imshow("Image", img)  # Display the captured frame in a window named "Image"


    if cv.waitKey(1) & 0xFF == ord('q'):  # Wait for the 'q' key to be pressed
        break  # Exit the loop if 'q' is pressed

cap.release()  # Release the camera
cv.destroyAllWindows()  # Close all OpenCV windows