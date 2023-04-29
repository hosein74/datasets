import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

qr_detector = cv2.QRCodeDetector()


def qr(path):
    for fn in Path(path).iterdir():
        img = cv2.imread(str(fn))
        x = qr_detector.detectAndDecodeMulti(img)
        coords = x[2].astype(int)
        names = x[1]
        color = np.array([255 // 3] * 3)
        print(f'{len(names)} names found. '
              f'{len(names) - names.count("")} readable:')
        for i, ((xl, yl), (xh, yh)) in enumerate(zip(coords.min(axis=1), coords.max(axis=1))):
            print(color, '\t', xl, yl, xh, yh, names[i])
            img[yl:yh, xl:xh] = (color + img[yl:yh, xl:xh]) // 2
            color[i % 3] = 255 - color[i % 3]
        print(' ')
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    qr('images')
