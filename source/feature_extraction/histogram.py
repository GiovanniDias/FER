import cv2

'''
https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
'''


class Histogram:
    def __init__(self, image=None):
        pass

    def faceHistogram(self, image, mask, bins=256):
        histogram = cv2.calcHist([image], [0], mask, [bins], [0, 256])
        return histogram

    def landmarksHistogram(self, image, mask_list, bins=256):
        histogramList = []
        for i, mask in enumerate(mask_list):
            histogramList += [cv2.calcHist([image],
                                           [0], mask, [bins], [0, 256])]
        return histogramList
