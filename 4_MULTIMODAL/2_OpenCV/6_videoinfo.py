import cv2
import sys

cap = cv2.VideoCapture('./movies/232538_tiny.mp4')

if not cap.isOpened():
    print('Cannot Load Video.')
    sys.exit()

print('Video loaded successfully.')

# 가로 픽셀 수 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Width: ', width)

# 세로 픽셀 수 가져오기
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Height: ', height)

# 전체 영상 프레임 수 가져오기
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame Count: ', frame_count)

# fps 출력
fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS: ', fps)

# 반복문은 기본적으로 CPU 속도로 계산하기 때문에 delay를 설정 해서 원본처럼 보이게 해야함.
# FPS를 이용해 프레임 사이 대기 시간을 계산
# 예: 30FPS라면 1초에 30장을 보여주므로 약 33ms마다 한 프레임을 표시함
delay = max(1, round(1000 / fps)) if fps > 0 else 33

while True: 
    # ret: 프레임을 정상적으로 읽었는지 여부 > T/F
    # frame: 읽어온 한 장의 영상 프레임 ndarray로 저장
    ret, frame = cap.read()  # 1프레임 가져오기
    # 영상 끝에 도달하거나 읽기에 실패하면 ret=False가 됨
    if not ret: 
        break
    cv2.imshow("frame", frame)
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
