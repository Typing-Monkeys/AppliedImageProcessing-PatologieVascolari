import cv2
import numpy as np
from os import walk, sep
import argparse


def register(ref_img: str, target_img: str):
    # Open the image files.
    img1_color = cv2.imread(target_img)  # Image to be aligned.
    img2_color = cv2.imread(ref_img)  # Reference image.
    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape
    # _, img1 = cv2.threshold(img1, 0, , cv2.THRESH_BINARY)

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(200)
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    # matches.sort(key = lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.98)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color, homography,
                                          (width, height))

    matches_key = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], None)
    cv2.imshow("matchese", matches_key)
    cv2.imshow("registered", transformed_img)

    cv2.waitKey(0)
    return transformed_img


def main(args: list[str]):
    parser = argparse.ArgumentParser(description='Auto Image Registraton')
    parser.add_argument('ref_dir', type=str, help='Ref images dir')

    parser.add_argument('target_dir', type=str, help='Target images dir')

    parser.add_argument('dest_dir',
                        type=str,
                        help='Folder where to store results')
    args = parser.parse_args()

    ref_list = []
    target_list = []

    for (dirpath, _, filenames) in walk(args.ref_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                img_full_path = sep.join([dirpath, filename])
                ref_list.append(img_full_path)

    for (dirpath, _, filenames) in walk(args.target_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                img_full_path = sep.join([dirpath, filename])
                target_list.append(img_full_path)

    ref_list.sort(
        key=lambda x: int(x.split('.')[0].split(sep)[-1].split('_')[1]))
    target_list.sort(
        key=lambda x: int(x.split('.')[0].split(sep)[-1].split('_')[1]))

    for i in range(len(ref_list)):
        # print(f"{ref_list[i]}----{target_list[i]}")

        out_img = register(ref_list[i], target_list[i])
        cv2.imwrite(sep.join([args.dest_dir, f"REGISTERED_{i}.png"]), out_img)

    # out_img = register()


if __name__ == "__main__":
    from sys import argv

    main(argv[1:])
