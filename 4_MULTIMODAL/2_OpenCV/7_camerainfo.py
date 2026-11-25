import cv2
import sys

# 0은 일반적으로 기본 웹캠을 의미
# 카메라가 여러 개라면 환경에 따라 1, 2 등의 번호를 사용할 수 있음
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cannot open cameara')
    sys.exit()

print('Camera connected successfully')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print('Width: ', width)
print('Height: ', height)
print('Camera FPS: ', fps)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame")
        break
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) == 27:  # ESC 눌러서 종료
        break

cap.release()
cv2.destroyAllWindows()