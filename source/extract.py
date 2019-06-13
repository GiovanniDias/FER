# external packages
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# internal packages
from feature_extraction.lbp import generateLBP
from feature_extraction.mask import getMasks
from feature_extraction.detection import Detector
from feature_extraction.histogram import Histogram


def preprocess(image):
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 10, 5)
    preprocessed = denoised
    return preprocessed


def concatenateHistogram(hists):
    scaler = MinMaxScaler()
    feature_vector = np.array([])
    for hist in hists:
        feature_vector = np.append(feature_vector, scaler.fit_transform(hist))
    return feature_vector


def getFeatureName(mark, bin):
    return 'f_{}_{}'.format(str(mark).zfill(2), str(bin).zfill(3))


def getFeatures(loaded_dataframe, bins = 32):
    # Instantiation
    detector = Detector()
    histogram = Histogram()

    feature_vectors = []
    for k, path in enumerate(loaded_dataframe['image_path']):
        print("Detectar Face na imagem {}.".format(k))
        image = preprocess(cv2.imread(path, 0))
        faces = detector.dlibDetection(image)
        for face in faces:
            '''
                GENERATE MASKS AND APPLY LBP TRANSFORM
            '''
            print("Geração de máscara e aplicação de LBP.")
            masks = getMasks(image, face, (0, 67))
            lbp = generateLBP(image, masks['face'])

            '''
                GENERATE HISTOGRAMS AND FEATURE VECTOR
            '''
            print("Geração de Histograma e Vetor de características.")
            hists = {'face': histogram.faceHistogram(lbp, masks['face'], bins),
                        'marks': histogram.landmarksHistogram(lbp, masks['marks'], bins)}
            features = concatenateHistogram(hists['marks'])
            feature_vectors += [{'emotion': loaded_dataframe['label'][k]}]
            # name features
            mark_index = 0
            item_index = len(feature_vectors) - 1
            for f, feature in enumerate(features):
                if f % bins == 0:
                    mark_index += 1
                key = getFeatureName(mark_index, f % bins)
                feature_vectors[item_index][key] = feature

    feature_dataframe = pd.DataFrame(feature_vectors)
    return feature_dataframe


if __name__ == '__main__':
    for i in range(0,1):
        bins = 16
        path1 = 'files/paths/paths_and_labels' + str(i) + '.csv'
        path2 = 'files/features/extracted_features' + str(i) + '_' + str(bins) + '.csv'
        df = pd.read_csv(path1)
        features = getFeatures(df, bins)
        features.to_csv(path2, index=False)
