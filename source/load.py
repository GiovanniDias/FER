import os
import cv2
import numpy as np
import imutils
import pandas as pd

root_path = '/home/giovanni/Documents'
images_path = root_path + '/cohn-kanade-images'
emotion_path = root_path + '/Emotion'


def getLabels():
    paths = []
    labels = []
    indexes = []
    for i, walker in enumerate(os.walk(emotion_path)):
        if walker[2]:
            paths += [os.path.abspath(walker[0]).split('Emotion/')[1]]
            indexes += [i]
            labels += [walker[2][0]]

    return paths, labels, indexes


def getImages(indexes):
    paths = []
    images = []
    for i, walker in enumerate(os.walk(images_path)):
        if i in indexes and walker[2]:
            paths += [os.path.abspath(walker[0]
                                      ).split('cohn-kanade-images/')[1]]
            images += [sorted(walker[2])]

    return paths, images


def getImagesData(n_images=[1, 4, 4, 3, 4, 4, 4, 4], all_images=False):
    label_paths, labels, indexes = getLabels()
    image_paths, images = getImages(indexes)
    data = {'image_path': [], 'label': []}
    for i, path in enumerate(label_paths):
        full_label_path = os.path.join(emotion_path, path, labels[i])
        file = open(full_label_path, 'r')
        try:
            label = int(float(file.read()))
            file.close()
        except:
            file.close()
            continue

        if all_images:
            for image in images[i]:
                if image != '.DS_Store':
                    full_image_path = os.path.join(
                        images_path, image_paths[i], image)
                    data['label'] += [label]
                    data['image_path'] += [full_image_path]
        else:
            if images[i][0] != '.DS_Store':
                full_image_path = os.path.join(
                    images_path, image_paths[i], images[i][0])
                data['label'] += [0]
                data['image_path'] += [full_image_path]

            image_set = images[i][-n_images[label]:]
            for j, image in enumerate(image_set):
                if image != '.DS_Store':
                    full_image_path = os.path.join(
                        images_path, image_paths[i], image)
                    data['label'] += [label]
                    data['image_path'] += [full_image_path]

    df = pd.DataFrame(data).sort_values(by=['image_path'])
    return df


if __name__ == '__main__':
    # OLD COMBINATION
    combinations = [
        [1, 4, 4, 3, 4, 4, 4, 4], [1, 4, 4, 3, 4, 2, 4, 2],
        [1, 2, 4, 2, 3, 2, 3, 2], [1, 2, 4, 2, 3, 1, 3, 1]
    ]
    # GET ONLY THE CLIMAX IMAGE
    combinations = [[1, 1, 1, 1, 1, 1, 1, 1]]
    for order in range(len(combinations)):
        csv = getImagesData(combinations[order])
        path = "files/paths/paths_and_labels" + str(order + 1) + ".csv"
        csv.to_csv(path, index=False)
