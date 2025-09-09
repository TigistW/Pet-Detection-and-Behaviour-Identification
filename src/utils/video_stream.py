import cv2


class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def read(self):
        ret, frame = self.cap.read()
        return ret, frame


    def release(self):
        self.cap.release()