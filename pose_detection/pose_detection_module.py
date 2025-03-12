import cv2 as cv
import mediapipe as mp
import time


class poseTracking():
    
    def __init__(self,static_image_mode=False,model_complexity=1,smooth_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=static_image_mode,model_complexity=model_complexity,smooth_landmarks=smooth_landmarks,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.landmarks = None
        
    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmList
    
    


if __name__ == "__main__":
    cap = cv.VideoCapture('/home/owais/MyProjects/computer_vision/pose_detection/run.mp4')
    pTime = 0
    detector = poseTracking()
    speed = 50  

    while True:
        success, img = cap.read()
        if not success:
            break  
        
        
        img = detector.findPose(img)
        pos = detector.findPosition(img)
        print(pos)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)

        key = cv.waitKey(speed)  
        if key == ord('q'):  
            break
        elif key == ord('+'):  
            speed = max(1, speed - 10)
        elif key == ord('-'):  
            speed += 10