from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
# bounding box : 물체의 경계를 딴 사각형
# hyper-parameters : 머신러닝 알고리즘을 위해 기술자가 지정하는 튜닝 option
frame_window = 10   # 크기가 10
emotion_offsets = (20, 40) # 튜플 자료형, window frame 안의 (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
# face_detection : 얼굴 감지
# load_detection_model : inference(추론).py에 있는 함수

# 출처_형준
# 모델의 형태를 감지 및 불러오는 함수(haarcascade의 경로)
# OpenCV 모듈 cv2의 CascadeClassifier 클래스를 이용해 비디오 스트림 객체 감지. 모델 경로를 설정하여 모델 트레이닝
# CascadeClassifier()로 opencv가 제공하는 사전 훈련된 분류기들을 사용할 수 있다.(예: 얼굴, 눈, 신체 등)

emotion_classifier = load_model(emotion_model_path, compile=False)
# emotion_classifier : 감정 분류
# load_model : tensorflow_keras 모듈
# emotion_model_path : 문자열 저장된 모델의 경로 - 모델 h5py.File을로드 할 객체
# compile=False : 컴파일 유무를 설정





# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
# input_shape : 모델의 입력 모양

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
# 창 식별자로 사용할 창의 이름 = window_frame
video_capture = cv2.VideoCapture(0)
# 비디오 캡쳐 객체를 생성함. 안의 숫자는 장치 인덱스(어떤 카메라를 사용할 것인가)임.
# 1개만 부착되어 있으면 0, 2개 이상이면 첫 웹캠은 0, 두번째 웹캠은 1으로 지정합니다
while True: # 특정 키를 누를 때까지 무한 사용
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # openCV는 기본적으로 BGR로 이미지를 import 해오기 때문에 cvtColor메소드로 RGB로 변경하여 변수에 저장함.
    faces = detect_faces(face_detection, gray_image)
    # image에 있는 사람의 이미지를 추적함.

    for face_coordinates in faces:  # faces 안에 좌표가 잡히는 동안

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        # 오프셋을 적용한뒤 리턴하는 함수
        # 좌표점에 특정 오프셋 값을 적용시킴으로써 값에 변화를 줌
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
