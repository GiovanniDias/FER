import dlib
import cv2
from imutils import face_utils
import numpy as np

'''
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
'''


class Detector:
    def __init__(
            self, cascade_path="source/models/lbpcascade_frontalface.xml",
            predictor_path="source/models/shape_predictor_68_face_landmarks.dat"):
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.cv_detector = cv2.CascadeClassifier(cascade_path)
        self.predictor = dlib.shape_predictor(predictor_path)

    def dlibDetection(self, image):
        detected = []
        rects = self.dlib_detector(image, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(image, rect)
            shape = face_utils.shape_to_np(shape)

            detected += [{'face': face_utils.rect_to_bb(rect),
                            'landmarks': shape}]

        return detected

    def cvDetection(self, image):
        detected_faces = self.cv_detector.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        detected = []
        for index, (x, y, w, h) in enumerate(detected_faces):
            # Converting the OpenCV rectangle coordinates to Dlib rectangle
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = self.predictor(image, dlib_rect).parts()

            landmarks = np.asarray(
                np.matrix([[p.x, p.y] for p in detected_landmarks])
            )

            detected += [{'face': detected_faces[index], 'landmarks': landmarks}]

        return detected