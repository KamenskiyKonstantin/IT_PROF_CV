import mediapipe
import cv2
import numpy as np
from time import sleep

cap = cv2.VideoCapture(0)
MODEL = mediapipe.solutions.hands.Hands(max_num_hands=2)

FINGER_BASES = list(range(1, 18, 4))

EPS = 50


class GestureDetector():
    def __init__(self, landmarks, shape, ) -> None:
        self.landmarks = landmarks
        self.shape = shape
        self.points = []

    def get_points(self):
        for mark in self.landmarks.landmark:
            self.points.append([mark.x * self.shape[1], mark.y * self.shape[0]])

    def rechognize(self):
        global EPS
        self.get_points()
        detected_gesture = ''
        cnt = 0
        for base in FINGER_BASES:
            if base == 1:
                EPS*=2
            if self.points[base][1] - EPS > self.points[base + 3][1]:
                cnt += 1
            if base == 1:
                EPS//=2


        res_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hands", img)

        detected_gesture = str(cnt)
        return detected_gesture



while cap.isOpened():
    ret, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    img = np.fliplr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = MODEL.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detector = GestureDetector(hand_landmarks, img.shape)

            print(detector.rechognize(), end='-')
        print()



