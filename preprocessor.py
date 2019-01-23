import numpy as np
from scipy.misc import imread, imresize      # python API이다
# SciPy는 PIL의 사용없이도 jpg and png 이미지를 즉각적으로 읽을수 있다.
# 또한 numpy arrays에 SciPy images가 저장이 된다,
# 사용자들은 직관적으로 데이터에 접근이 가능하다. 제외하고 visualization.

def preprocess_input(x, v2=True):
    x = x.astype('float32')         # 인자 값을 float형으로 바꿀때 사용
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
# keras 라이브러리 내재된 메소드
# 또한 모델에 필요한 형식으로 이미지를 적절히 표시하기위한 것

def _imread(image_name):
            return imread(image_name)
# 저장된 이미지를 리턴, cv2에서 이미지를 읽는 방법중 하나.

def _imresize(image_array, size):
        return imresize(image_array, size)
# 미지의 크기를 변경하기 위한 메소드

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
# tf.keras.utils.to_categorical는 클래스 벡터(정수)를 이진 클래스 행렬도 변환
# 첫번째 인자는 클래스 벡터가 행렬로 변환
# 클래스의 총 갯수를 의미
