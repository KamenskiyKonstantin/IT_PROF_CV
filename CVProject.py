import mediapipe
import cv2
import numpy as np
from time import sleep

cap = cv2.VideoCapture(0)
MODEL = mediapipe.solutions.hands.Hands(max_num_hands=2)

FINGER_BASES = list(range(1, 18, 4))

EPS = 50

GESTURE_CONFIDENCE = 5

confidence_gesture = ''
cur_gesture = ''

confidence = 0


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

        if (self.points[4][1] < self.points[3][1]
            and self.points[5][1] < self.points[9][1]
            and self.points[8][0] > self.points[7][0]
            and self.points[12][0] > self.points[11][0]
            and self.points[16][0] > self.points[15][0]
            and self.points[20][0] > self.points[19][0]
            and self.points[8][0] < self.points[0][0]
            and self.points[12][0] < self.points[0][0]
            and self.points[16][0] < self.points[0][0]
            and self.points[20][0] < self.points[0][0]):

            return '='

        cnt = 0
        for base in FINGER_BASES:
            if self.points[base][1] - EPS > self.points[base + 3][1]:
                cnt += 1
                if base == 1 and abs(self.points[base][0] - self.points[base + 3][0]) > 1.5 * EPS:
                    cnt -= 1

        return str(cnt)


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
            cur_gesture += detector.rechognize()
        if '=' in cur_gesture:
            cur_gesture = '='

        if cur_gesture != confidence_gesture:
            confidence_gesture = cur_gesture
            confidence = 1
        else:
            confidence += 1

        if confidence >= GESTURE_CONFIDENCE:
            print(cur_gesture, 'CONFIDENT')

        cur_gesture = ''

    cv2.imshow("Hands", img)
