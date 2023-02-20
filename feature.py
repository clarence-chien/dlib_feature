import cv2
import dlib

# 載入人臉偵測器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 載入特徵點檢測器
facial_landmark_detector = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')

# 讀取圖片
img = cv2.imread('test.jpg')

# 轉為灰階圖片，方便進行偵測
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 偵測人臉位置
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 用綠色方框標示出所有偵測到的人臉位置
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 偵測人臉特徵點
for (x, y, w, h) in faces:
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = facial_landmark_detector(gray, rect)
    for i, point in enumerate(landmarks.parts()):
        x, y = point.x, point.y
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

# 顯示處理後的圖片
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
