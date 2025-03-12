import cv2 as cv 
import mediapipe as mp  
import time

class handDetection():

    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=static_image_mode, 
            max_num_hands=max_num_hands, 
            model_complexity=model_complexity,  
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraws = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handlems in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraws.draw_landmarks(img, handlems, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def hand_landmark(self, index_no=0, draw=True):
        
        if self.results.multi_hand_landmarks:
            for handlems in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handlems.landmark):
                    height, weight, channnels = img.shape
                    
                    if id == index_no:
                        cx, cy = int(lm.x*weight), int(lm.y*height)
                        
                        
                        if draw:
                            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
                        
                        return (cx, cy, lm)            
        return None



if __name__ == "__main__":

    cap = cv.VideoCapture(0)
    
    hand_detector = handDetection()
    
    pTime = 0

    while True:
        success, img = cap.read()

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert the BGR image to RGB
        
        
        img = hand_detector.findHands(img)
        position= hand_detector.hand_landmark(index_no=15)
        
        if position:
            print(position)
            
            
            
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv.imshow("Image", img)  


        if cv.waitKey(1) & 0xFF == ord('q'):  
            break  
