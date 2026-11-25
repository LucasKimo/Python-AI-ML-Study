import cv2
import numpy as np

"""
투시 변환(Perspective Transfrom)
투시 변환은 원근감 때문에 기울어져 보이는 사각형 영역을 정면에서 바라본 것처럼 펴는 데 사용됨
"""

img = cv2.imread("./images/pic.jpg")
dst_w = 600
dst_h = 400

# 원본 영상에서 선택한 책 꼭지점 좌표 
src_quad = np.array([
    # 왼쪽 위,    오른쪽 위,    오른쪽 아래,   왼쪽 아래
    [370, 173], [1224, 157], [1417, 830], [209, 849]
], dtype=np.float32)

# 변환 후 꼭지점 좌표
dst_quad = np.array([
    [0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]
], dtype=np.float32)

# getPerspectiveTransform(): 네 쌍의 대응점을 이용해 3*3 투시 변환 행렬을 계산
# matrix 저장
perspective_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
# 변환함수에 matrix 적용
dst = cv2.warpPerspective(img, perspective_matrix, (dst_w, dst_h))

preview = img.copy()
# 각 꼭지점에 원 그리기
for pt in src_quad.astype(int):
    # cv2.circle(이미지, 좌표, 반지름, 색상, 안에 색상 채우기)
    cv2.circle(preview, tuple(pt), 8, (0, 0, 255), -1)

# 꼭지점 선으로 연결하기
# cv2.polylines(이미지, 좌표, 연결, 색상, 두께)
cv2.polylines(preview, [src_quad.astype(np.int32)], True, (0, 255, 0), 3)

cv2.imshow("source", preview)
cv2.imshow('result', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()