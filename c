import cv2
import numpy as np
import time
import mediapipe as mp
import sys
from mode import voice_mode

# Settings and initializations
camera = cv2.VideoCapture(0)
settings = {
    'window_width': 1280,
    'window_height': 720,
    'color_swatches': [
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 0)
    ],
    'brush_size': {'Small': 5, 'Medium': 15, 'Large': 25},
    'drawState': ['Draw', 'Erase', 'Standby'],
}
color_idx = list(range(len(settings['color_swatches'])))
brush_size = settings['brush_size']['Medium']
color = 0
drawState = 'Standby'
prevcanvas = np.zeros((settings['window_height'], settings['window_width'], 3), dtype=np.uint8)
run = True
fps = 0
fpsfilter = 0.9
savetime = -1

# Mediapipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition data
keypoints = [0, 5, 9, 13, 17, 8, 12, 16, 20]
threshold = 1.2
knowngestures = []
gesturenames = ['Small', 'Medium', 'Large', 'Draw', 'Erase', 'Standby']

# Load gesture data
# Placeholder: Replace with actual gesture loading logic
def load_gesture_data():
    global knowngestures
    # Load gesture data logic here
    pass

load_gesture_data()

# Helper functions
def findDistances(hand):
    points = []
    for id in keypoints:
        points.append((hand.landmark[id].x, hand.landmark[id].y))
    distances = np.zeros((len(points), len(points)), dtype=np.float32)
    for i in range(len(points)):
        for j in range(len(points)):
            distances[i][j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    return distances

def findError(distances, gesture, keypoints):
    error = 0
    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            error += abs(distances[i][j] - gesture[i][j])
    return error / (len(keypoints) ** 2)

def findhands(image):
    return hands.process(image)

def preprocess(frame, drawState, fps):
    cv2.putText(frame, f'Mode: {drawState}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def detect_gesture(frame, hand_data):
    if len(hand_data) > 0:
        distance_matrix = findDistances(hand_data[0])
        error = []
        for i in range(len(knowngestures)):
            error.append(findError(distance_matrix, knowngestures[i], keypoints))
        error = np.array(error)
        best_match = np.argmin(error)
        if error[best_match] < threshold:
            return gesturenames[best_match]
    return None

def handle_gesture(gesture):
    global color, brush_size, drawState
    if gesture in color_idx:
        color = gesture
    elif gesture in settings['brush_size']:
        brush_size = settings['brush_size'][gesture]
    elif gesture in settings['drawState']:
        drawState = gesture

def draw(frame, hand_data, color, brush_size):
    if len(hand_data) > 0:
        x, y = hand_data[0][8][0], hand_data[0][8][1]
        if 0 <= y <= 60:
            if 0 <= x < settings['window_width']//10:
                color = color_idx[0]
            elif settings['window_width']//10 <= x < 2*settings['window_width']//10:
                color = color_idx[1]
            elif 2*settings['window_width']//10 <= x < 3*settings['window_width']//10:
                color = color_idx[2]
            elif 3*settings['window_width']//10 <= x < 4*settings['window_width']//10:
                color = color_idx[3]
            elif 4*settings['window_width']//10 <= x < 5*settings['window_width']//10:
                color = color_idx[4]
            elif 5*settings['window_width']//10 <= x < 6*settings['window_width']//10:
                color = color_idx[5]
            elif 6*settings['window_width']//10 <= x < 7*settings['window_width']//10:
                color = color_idx[6]
            elif 7*settings['window_width']//10 <= x < 8*settings['window_width']//10:
                color = color_idx[7]
            elif 8*settings['window_width']//10 <= x < 9*settings['window_width']//10:
                color = color_idx[8]
            elif 9*settings['window_width']//10 <= x < settings['window_width']:
                color = color_idx[9]
        elif settings['window_height']-60 <= y < settings['window_height']:
            if 0 <= x < settings['window_width']//8:
                drawState = 'Standby'
            elif settings['window_width']//8 <= x < 2*settings['window_width']//8:
                drawState = 'Draw'
            elif 2*settings['window_width']//8 <= x < 3*settings['window_width']//8:
                drawState = 'Erase'
            elif 3*settings['window_width']//8 <= x < 4*settings['window_width']//8:
                clearcanvas()
            elif 4*settings['window_width']//8 <= x < 5*settings['window_width']//8:
                global savetime
                savetime = time.time()
            elif 5*settings['window_width']//8 <= x < 6*settings['window_width']//8:
                cv2.destroyAllWindows()
                sys.exit()
        else:
            if drawState == 'Draw':
                cv2.circle(prevcanvas, (x, y), brush_size, settings['color_swatches'][color], -1)
            elif drawState == 'Erase':
                cv2.circle(prevcanvas, (x, y), brush_size, (0, 0, 0), -1)
    return prevcanvas

while run:
    ret, frame = camera.read()
    if not ret:
        break
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = findhands(framergb)
    hand_data = results.multi_hand_landmarks

    gesture = detect_gesture(frame, hand_data)
    if gesture:
        handle_gesture(gesture)

    frame = preprocess(frame, drawState, fps)
    prevcanvas = draw(frame, hand_data, color, brush_size)
    frame = cv2.addWeighted(frame, 0.5, prevcanvas, 0.5, 0)

    endtime = time.time()
    fps = 1/(endtime-starttime)
    fps = (fpsfilter*fps + (1-fpsfilter)*fps)
    starttime = endtime

    if savetime != -1 and time.time() - savetime > 2:
        imgname = f'drawing_{int(savetime)}.png'
        cv2.imwrite(imgname, prevcanvas)
        savetime = -1

    cv2.imshow('OpenCV Paint', frame)
    key = cv2.waitKey(1)
    if key == ord('m'):
        voice_mode()

camera.release()
cv2.destroyAllWindows()