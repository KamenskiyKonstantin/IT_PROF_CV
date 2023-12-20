import mediapipe
import cv2
import numpy as np
from time import time_ns
import pygame
import sys
import

cap = cv2.VideoCapture(0)
MODEL = mediapipe.solutions.hands.Hands(max_num_hands=2)

EPS = 20

GESTURE_CONFIDENCE = 10
TIMER = 3000


def islower(a, b):
    return a < b and abs(a - b) > EPS


def ishigher(a, b):
    return a > b and abs(a - b) > EPS


def isequal(a, b):
    return abs(a - b) < EPS

class GestureDetector():
    def __init__(self, landmarks, shape, hand) -> None:
        self.landmarks = landmarks
        self.shape = shape
        self.points = []
        self.hand = hand

    def get_points(self):
        for mark in self.landmarks.landmark:
            self.points.append([mark.x * self.shape[1], mark.y * self.shape[0]])

    def rechognize(self):
        global EPS
        self.get_points()

        if (self.points[4][1] < self.points[3][1]

                and ishigher(self.points[8][0], self.points[7][0])
                and ishigher(self.points[12][0], self.points[11][0])
                and ishigher(self.points[16][0], self.points[15][0])
                and ishigher(self.points[20][0], self.points[19][0])
                and islower(self.points[8][0], self.points[0][0])
                and islower(self.points[12][0], self.points[0][0])
                and islower(self.points[16][0], self.points[0][0])
                and islower(self.points[20][0], self.points[0][0])
                and self.points[5][1] < self.points[9][1]):

            return '='
        elif abs(self.points[5][1] - self.points[8][1]) < EPS < abs(self.points[5][0] - self.points[8][0]):
            if abs(self.points[9][1] - self.points[12][1]) < EPS < abs(
                    self.points[9][0] - self.points[12][0]):
                return 'backspace'
            return '-'
        elif (
                islower(self.points[8][1], self.points[5][1])
                and ishigher(self.points[12][1], self.points[10][1])
                and ishigher(self.points[16][1], self.points[14][1])
                and ishigher(self.points[20][1], self.points[18][1])
        ):
            return '1'
        elif (
                islower(self.points[8][1], self.points[5][1])
                and islower(self.points[12][1], self.points[9][1])
                and ishigher(self.points[16][1], self.points[14][1])
                and ishigher(self.points[20][1], self.points[18][1])
        ):
            return '2'
        elif (
                islower(self.points[8][1], self.points[5][1])
                and islower(self.points[12][1], self.points[9][1])
                and islower(self.points[16][1], self.points[13][1])
                and ishigher(self.points[20][1], self.points[18][1])
        ):
            return '3'
        elif (
                islower(self.points[8][1], self.points[5][1])
                and islower(self.points[12][1], self.points[9][1])
                and islower(self.points[16][1], self.points[13][1])
                and islower(self.points[20][1], self.points[17][1])

        ):

            if 'Left' in str(self.hand):
                if self.points[4][0] < self.points[1][0]:
                    return '4'
                else:
                    return '5'
            else:
                if self.points[4][0] > self.points[1][0]:
                    return '4'
                else:
                    return '5'
        elif (
                ishigher(self.points[8][1], self.points[6][1])
                and ishigher(self.points[12][1], self.points[10][1])
                and ishigher(self.points[16][1], self.points[14][1])
                and ishigher(self.points[20][1], self.points[18][1])
        ):
            return '0'
        return 'NONE'


confidence_gesture = ''
cur_gesture = ''
confidence = 0

expr = ''

current_time = 0





while cap.isOpened():
    ret, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break
    img = np.fliplr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = MODEL.process(img)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_base = int(hand_landmarks.landmark[0].x * img.shape[1])
            y_base = int(hand_landmarks.landmark[0].y * img.shape[0])
            cv2.circle(img, (x_base, y_base), radius=10, color=(0, 0, 255), thickness=-1)


    if time_ns()//1000000 - current_time > TIMER:

        results = MODEL.process(img)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                detector = GestureDetector(hand_landmarks, img.shape, results.multi_handedness[i])
                cur_gesture += detector.rechognize()

            if cur_gesture != '':

                if 'NONE' in cur_gesture:
                    cur_gesture = ''
                if '1' in cur_gesture and '-' in cur_gesture:
                    cur_gesture = '+'
                elif '-' in cur_gesture:
                    cur_gesture = '-'
                elif '=' in cur_gesture:
                    cur_gesture = '='
                elif 'backspace' in cur_gesture:
                    cur_gesture = 'backspace'
                elif cur_gesture.isdigit():
                    cur_gesture = sum(list(map(int, list(cur_gesture))))
                    if cur_gesture == 10:
                        cur_gesture = '0'
                if cur_gesture != confidence_gesture:
                    confidence_gesture = cur_gesture
                    confidence = 1
                else:
                    confidence += 1

                if cur_gesture != '':
                    print('detecting finished... gesture rechognized as', cur_gesture, 'confidence:', confidence, '/', GESTURE_CONFIDENCE)
                else:
                    print('detecting finished... gesture rechognized as', 'unknown gesture', 'confidence:', confidence, '/',
                          GESTURE_CONFIDENCE)

                if confidence >= GESTURE_CONFIDENCE:
                    print('proceed')
                    confidence_gesture = ''
                    confidence = 0
                    if cur_gesture == '=':
                        try:
                            print('your expression was', expr, 'your result is', eval(expr))
                        except SyntaxError:
                            print('your expression was', expr, 'Invalid expression')
                        except ZeroDivisionError:
                            print('your expression was', expr,'Zero division')
                        cur_gesture = ''
                        expr = ''

                    elif cur_gesture == 'backspace':
                        expr = expr[:-1]
                    else:
                        expr += str(cur_gesture)
                        try:
                            if expr[-1] == expr[-2] == '+':
                                expr = expr[:-2]+'*'
                            elif expr[-1] == expr[-2] == '-':
                                expr = expr[:-2]+'/'
                        except IndexError:
                            pass

                    current_time = time_ns()//1000000
                cur_gesture = ''
            else:
                print('unknown gesture')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", img)
