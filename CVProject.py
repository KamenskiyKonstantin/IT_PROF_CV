import mediapipe
import cv2
import numpy as np
from time import time_ns
import pygame
import os
from Constants import CAMERA_ID, TIMER, GESTURE_CONFIDENCE, FONT

EPS = 30


"""
Нижеследующие три функции сравнивают числа с учетом значения EPS
"""
def islower(a:float, b:float) -> bool:
    return a < b and abs(a - b) > EPS


def ishigher(a:float, b:float) -> bool:
    return a > b and abs(a - b) > EPS


def isequal(a:float, b:float) -> bool:
    return abs(a - b) < EPS


class GestureDetector:
    """
    Represents a detector of gestures on the image
    """

    def __init__(self, landmarks, shape: (float, float), hand) -> None:
        """
        initializes recognizers
        :param landmarks: Normalized landmark list, list of landmarks of hand
        :param shape: Input image shape
        :param hand: data about left or right hand is provided
        """
        self.landmarks = landmarks
        self.shape = shape
        self.points = []
        self.hand = hand

    def get_points(self) -> None:
        """
        defines list of key points on landmark
        :return:
        """
        for mark in self.landmarks.landmark:
            self.points.append([mark.x * self.shape[1], mark.y * self.shape[0]])

    def recognize(self) -> str:
        """
        detects gesture and returns it as a string
        :return:
        """
        global EPS
        self.get_points()




        if (
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
        elif not isequal(self.points[5][0], self.points[8][0]):
            if not isequal(
                    self.points[9][0], self.points[12][0]):
                if not isequal(
                    self.points[13][0], self.points[16][0]):
                    return '='
                return 'backspace'
            return '-'
        return 'NONE'


def draw() -> None:
    """
    Draws the current parameters: detected gesture, confidence, current expression
    :return:
    """
    global cur_gesture
    pygame.draw.rect(screen, (17, 17, 17), screen.get_rect())
    if cur_gesture == '':
        cur_gesture = 'Unknown'

    gesture_label = main_font.render(f'Detected gesture: {cur_gesture}', True, (238, 238, 238))
    screen.blit(gesture_label, (10, 10))

    if cur_gesture == 'Unknown':
        cur_gesture = ''

    confidence_label = main_font.render(f'Confidence: {confidence}/{GESTURE_CONFIDENCE}', True,
                                        (238, 238, 238))
    screen.blit(confidence_label, (10, 30))

    cur_expr_label = main_font.render(f'Current entered expression: {expr}', True, (238, 238, 238))
    screen.blit(cur_expr_label, (10, 50))


MODEL = mediapipe.solutions.hands.Hands(max_num_hands=2)



cap = cv2.VideoCapture(CAMERA_ID)

confidence_gesture = ''
cur_gesture = ''
confidence = 0

expr = ''

current_time = 0

pygame.init()
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (650, 30)
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption('Results')

pygame.font.init()
main_font = pygame.font.SysFont(FONT, 14, False, False)

while cap.isOpened():
    ret, img = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
    except pygame.error:
        break

    img = np.fliplr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = MODEL.process(img)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_base = int(hand_landmarks.landmark[0].x * img.shape[1])
            y_base = int(hand_landmarks.landmark[0].y * img.shape[0])
            cv2.circle(img, (x_base, y_base), radius=10, color=(0, 0, 255), thickness=-1)

    if time_ns() // 1000000 - current_time > TIMER:

        results = MODEL.process(img)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                detector = GestureDetector(hand_landmarks, img.shape, results.multi_handedness[i])
                cur_gesture += detector.recognize()

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

                draw()

                pygame.display.flip()

                if cur_gesture != '':
                    print('detecting finished... gesture rechognized as', cur_gesture, 'confidence:', confidence, '/',
                          GESTURE_CONFIDENCE)
                else:
                    print('detecting finished... gesture rechognized as', 'unknown gesture', 'confidence:', confidence,
                          '/',
                          GESTURE_CONFIDENCE)

                if confidence >= GESTURE_CONFIDENCE:
                    print('proceed')
                    confidence_gesture = ''
                    confidence = 0
                    if cur_gesture == '=':
                        try:
                            res = eval(expr)
                        except SyntaxError:
                            res = 'Invalid expression'
                        except ZeroDivisionError:
                            res = 'Zero division'

                        print(f'your expression was {expr},', 'your result is', res)

                        current_time = time_ns() // 1000000

                        draw()

                        result_lbl = main_font.render(f'Result: {res}', True, (238, 238, 238))
                        screen.blit(result_lbl, (10, 70))

                        pygame.display.flip()

                        cur_gesture = ''
                        expr = ''

                        continue

                    elif cur_gesture == 'backspace':
                        expr = expr[:-1]
                    else:
                        expr += str(cur_gesture)
                        try:
                            if expr[-1] == expr[-2] == '+':
                                expr = expr[:-2] + '*'
                            elif expr[-1] == expr[-2] == '-':
                                expr = expr[:-2] + '/'
                        except IndexError:
                            pass

                    draw()

                    current_time = time_ns() // 1000000
                cur_gesture = ''
            else:
                print('unknown gesture')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", img)
    try:
        pygame.display.flip()
    except pygame.error:
        break

MODEL.close()