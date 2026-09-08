import cv2 as cv, time
for idx in (0, 2, 3):
    cap = cv.VideoCapture(idx, cv.CAP_V4L2)
    opened = cap.isOpened()
    ok = False
    for i in range(20):
        ret, f = cap.read()
        if ret and f is not None and f.size > 0:
            print(idx, 'opened', opened, 'frame', i, f.shape)
            ok = True
            break
    if not ok:
        print(idx, 'opened', opened, 'NO FRAME in 20 reads')
    cap.release()
