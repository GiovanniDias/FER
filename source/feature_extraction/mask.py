import cv2
import numpy as np


def getMasks(image, face, landmarks_index=(0, 67)):
    first_landmark = landmarks_index[0]
    last_landmark = landmarks_index[1]

    mask_size = int(face['face'][-1] * 0.05)
    face_mask, masks = generateMask(
        image, face['landmarks'], mask_size, 'mask',
        'circle', first_landmark, last_landmark, True
    )
    return {'face': face_mask, 'marks': masks}


def generateMask(image, landmarks, size, name='mask', shape='rect',
                 initial_mark=0, final_mark=67, write_file=False):
    face_mask = np.zeros((image.shape), dtype="uint8")
    mark_mask = []
    for index, mark in enumerate(landmarks):
        mask = np.zeros((image.shape), dtype="uint8")
        if(index >= initial_mark) & (index <= final_mark):
            if shape == 'rect':
                top_left = tuple(
                    (mark[0] - size, mark[1] - size))
                bottom_righ = tuple(
                    (mark[0] + size, mark[1] + size))
                # draw piece mask
                cv2.rectangle(mask, top_left, bottom_righ, (255, 255, 255), -1)
                # draw full mask
                cv2.rectangle(face_mask, top_left,
                              bottom_righ, (255, 255, 255), -1)

            if shape == 'circle':
                cv2.circle(mask, tuple(mark), size, (255, 255, 255), -1)
                cv2.circle(face_mask, tuple(mark), size, (255, 255, 255), -1)

            mark_mask += [mask]

    if write_file:
        writeMasks(name, face_mask, mark_mask)

    return face_mask, mark_mask


def writeMasks(name, full_mask, piece_masks):
    file_name = 'full' + name + '.jpg'
    cv2.imwrite(file_name, full_mask)
    for index, mask in enumerate(piece_masks):
        file_name = name + str(index) + '.jpg'
        cv2.imwrite(file_name, mask)


def drawFaceBox(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def drawLandmarks(image, landmarks, text=False):
    for index, mark in enumerate(landmarks):
        cv2.circle(image, tuple(mark), 3, color=(0, 255, 255))
        if text:
            cv2.putText(image, str(index), tuple(mark),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))


if __name__ == '__main__':
    from detection import Detector
    detector = Detector()

    image = cv2.imread("images/test/face.jpg")
    faces = detector.dlibDetection(image)
    for face in faces:
        masks = getMasks(image, face, (0, 67),)

    cv2.imshow("Masked", image)
    cv2.waitKey(0)
