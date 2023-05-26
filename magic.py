import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import walk, sep
import argparse


def process(fpath: str, low=200, upper=255, verbose=False, iter=5, ksize=5):
    img = cv2.imread(fpath)

    if verbose:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title('Original Img')

    # in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sogliatura e binarizzazione
    _, thresh = cv2.threshold(gray, low, upper, cv2.THRESH_BINARY)

    if verbose:
        plt.figure()
        plt.imshow(thresh, cmap='gray')
        plt.title('Thresholded Img')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    thresh = cv2.medianBlur(thresh, ksize=5)

    eroded = cv2.erode(thresh, kernel=kernel, iterations=iter)

    # individuo le componenti connesse
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(
        eroded)

    output = np.zeros(thresh.shape, dtype="uint8")

    areas = []

    # Loop through each component
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        x, y = thresh.shape
        xc, yc = tuple(centroid[i])

        diffx = abs(x / 2 - xc)
        diffy = abs(y / 2 - yc)

        areas.append((area, i, diffx + diffy))

    # max_area, id = max(areas, key=lambda x: x[0])
    _, id, _ = min(areas, key=lambda x: x[2])

    # if area_low < area:
    #     # Labels stores all the IDs of the components on the each pixel
    #     # It has the same dimension as the threshold
    #     # So we'll check the component
    #     # then convert it to 255 value to mark it white
    componentMask = (label_ids == id).astype("uint8") * 255

    #     # Creating the Final output mask
    output = cv2.bitwise_or(output, componentMask)

    output = cv2.dilate(output, kernel=kernel, iterations=iter)

    if verbose:
        plt.figure()
        plt.imshow(output, cmap='gray')
        plt.title('Output Img')

    return output


def main(args: list[str]):
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('source_dir',
                        type=str,
                        help='A required integer positional argument')

    parser.add_argument('dest_dir',
                        type=str,
                        help='A required integer positional argument')

    # Optional argument
    parser.add_argument('--lower',
                        type=int,
                        default=190,
                        help='An optional integer argument')

    parser.add_argument('--upper',
                        type=int,
                        default=255,
                        help='An optional integer argument')
    parser.add_argument('--iter',
                        type=int,
                        default=2,
                        help='An optional integer argument')

    parser.add_argument('--ksize',
                        type=int,
                        default=5,
                        help='An optional integer argument')

    args = parser.parse_args()

    for (dirpath, _, filenames) in walk(args.source_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                img_full_path = sep.join([dirpath, filename])
                out_img = process(img_full_path,
                                  args.lower,
                                  args.upper,
                                  iter=args.iter,
                                  ksize=args.ksize)

                cv2.imwrite(sep.join([args.dest_dir, f"MASK_{filename}"]),
                            out_img)


if __name__ == "__main__":
    from sys import argv

    main(argv[1:])
