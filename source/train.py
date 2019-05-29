import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def select(features, rate=16/21):
    pca = PCA(n_components=90)
    selected = pca.fit_transform(features)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(selected.shape)
    return selected


def validate(train_result, test_result, predicted, y_test):
    print("TRAIN SCORE = ", train_result)
    print("TEST SCORE = ", test_result)
    print("F1 = ", f1_score(y_test, predicted))
    print("PRECISION = ", precision_score(y_test, predicted))
    print("RECALL = ", recall_score(y_test, predicted))
    print("ACCURACY = ", accuracy_score(y_test, predicted))
    print("CONFUSION MATRIX\n", confusion_matrix(y_test, predicted))


def train(features, reduce_dimension=True, only_eyes_mouth_nose=False):
    y = features['emotion'].values
    X = features.drop(['emotion'], axis=1)

    if only_eyes_mouth_nose:
        remove_cols = [str('f_' + str(i).zfill(2)) for i in range(1, 18)]
        cols = [c for c in X.columns if c[:4] not in remove_cols]
        X = X[cols]

    if reduce_dimension:
        print("Selecting...")
        # scaled = select(scaled, 18/21)
        scaled = select(scaled)
        print("Selected")

    svm = SVC(gamma=0.001)
    size = 0.25
    predicted = []

    print("Starting Training...")
    for i in range(int(1/size)):
        X_train, X_test, y_train, y_test = train_test_split(
            scaled, y, test_size=size, random_state=i, stratify=y)
        svm.fit(X_train, y_train)

        train_result = svm.score(X_train, y_train)
        test_result = svm.score(X_test, y_test)
        predicted = svm.predict(X_test)

        print(test_result)

        # validate(train_result, test_result, predicted, y_test)

    return test_result


if __name__ == '__main__':
    features = pd.read_csv('files/features/extracted_features1_32.csv')
    print("Features Loaded")
    model = train(features)
    print(model)
