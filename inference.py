import cv2 # opencv 모듈 가져오기

import matplotlib.pyplot as plt
import numpy as np
# pylabpyplot을 numpy와 결합하여 단일 네임 스페이스로 만듬.
# 차트와 데이터를 시각화해주는 패키지.

from keras.preprocessing import image
# 실시간 이미지 데이터 배치 생성 및 반복 처리

# *컬러 사진을 opencv에서는 BGR순서로 저장하나 matplotlib에서는 RGB순으로 저장함. 따라서 mat을 사용할땐 RGB로 변환하고 다시 opencv함수를 사용할때 BGR로 변환해야함.

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)
# 이미지 불러오는 함수(이미지 경로, 회색조 유무(기본값 False), 이미지 크기 설정(기본값 None 아닐때는 튜플로 수평 수직값))

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model
# 모델의 형태를 감지 및 불러오는 함수(haarcascade의 경로)
# OpenCV 모듈 cv2의 CascadeClassifier 클래스를 이용해 비디오 스트림 객체 감지. 모델 경로를 설정하여 모델 트레이닝
# CascadeClassifier()로 opencv가 제공하는 사전 훈련된 분류기들을 사용할 수 있다.(예: 얼굴, 눈, 신체 등)

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)
# 얼굴 형태 감지 및 사각형 도형을 시각화 표현하는 함수(얼굴 객체가 담긴 cascade객체, 객체가 담긴 회색조 이미지 배열)
# 회색으로 변환한 모델을 1.3의 스케일 팩터(이미지 크기를 줄이는 계수, 이미지의 크기를 줄이면 비용이 많이 들지만 정확도 상승)로 지정. minNeighbors가 5(각 후보 사각형을 유지해야하는 이웃 수를 지정하는 매개 변수. 값이 클수록 탐지는 줄지만 품질 상승)
# 리턴값으로 사각형의 Rect(x, y, w, h)값 반환 얼굴을 트레이닝 했으므로 얼굴의 좌표가 반환됨

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x,y), (x + w, y + h), color, 2)
# 감지된 모델을 사각형 도형으로 그리는 함수(얼굴을 가리키는 사각형 좌표, 이미지 저장 배열(비디오 캡쳐후 rgb로 바꾼 이미지), 색)
# cv2.rectangle은 opencv에서 제공하는 사각형 그리는 함수
# 1번째 인자 image_array는 이미지가 저장되있는 배열로 그림을 그릴 이미지
# 2번째 인자는 시작 좌표, 3번째 종료 좌표
# 4번째 BGR형태 컬러 설정, 5번째 선의 두께 설정

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x = x_off, x + width + x_off, y - y_off, y + height + y_off)
# 오프셋을 적용한뒤 리턴하는 함수
# 좌표점에 특정 오프셋 값을 적용시킴으로써 값에 변화를 줌

def darw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                 font_scale, color, thickness, cv2.LINE_AA)
#  사각형 위에 텍스트 그리는 함수(얼굴을 가리키는 사각형 좌표, rgb변환 이미지, 제목입력, x오프셋, y오프셋, 글씨크기, 글씨 두께)
#  x, y는 회색의 얼굴을 가리키는 사각형 좌표 리스트의 1,2번째 값으로 지정
#  cv2.putText(문자열을 그릴 이미지, 문자열, 문자열의 왼쪽 위 좌표, 폰트타입(normal size sans-serif font), 폰트 기본 크기에 곱해질 폰트 스케일 팩터, 글자 색, 글자 두께(디폴트 값1), cv2.LINE_AA는 좋게 보기위해 권장되는 값)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors
# 색 가져오는 함수
# np.linspace(선형 구간 혹은 로그 구간을 지정한 구간의 수만큼 분할(시작, 끝(포함), 분할 개수).tolist()
# * tolist()는 다차원 배열을 리스트로 변환하여 순서대로 나열한다.
#list로 저장된 colors를 np.asarray()를 이용해 배열로 변환 후 255를 곱하여 저장 뒤 리턴
