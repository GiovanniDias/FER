import numpy as np

'''
https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
'''
'''
http://www.sersc.org/journals/IJAST/vol36/5.pdf
'''

'''
http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_multiblock_local_binary_pattern.html#sphx-glr-auto-examples-features-detection-plot-multiblock-local-binary-pattern-py
'''

'''
http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html#sphx-glr-auto-examples-features-detection-plot-local-binary-pattern-py
'''


def lbpThreshold(pixel, neighbor):
    dif = int(neighbor) - int(pixel)
    if dif >= 0:
        return '1'
    return '0'


def lbpEncode(mat, row, col):
    code = ''
    for x in range(col - 1, col + 1):
        code += lbpThreshold(mat[row][col], mat[row - 1][x])

    for x in range(row - 1, row + 1):
        code += lbpThreshold(mat[row][col], mat[x][col + 1])

    for x in range(col + 1, col - 1, -1):
        code += lbpThreshold(mat[row][col], mat[row + 1][x])

    for x in range(row + 1, row - 1, -1):
        code += lbpThreshold(mat[row][col], mat[x][col - 1])

    return code


def generateLBP(image, mask, init=[1, 1]):
    rows, cols = image.shape
    lbp = np.zeros((rows, cols), dtype="uint8")
    for row in range(init[0], rows - 1):
        for col in range(init[1], cols - 1):
            if mask[row][col] != 0:
                encoded_value = lbpEncode(image, row, col)
                center_pixel = int(encoded_value, 2)
                lbp[row][col] = center_pixel

    return [lbp]


if __name__ == '__main__':
    import cv2
    from imutils import resize
    image = resize(cv2.imread("images/test/many_faces.jpg", 0), height=500)
    lbp = generateLBP(image, np.ones(image.shape))
    cv2.imshow("LBP", lbp[0])
    cv2.waitKey(0)
