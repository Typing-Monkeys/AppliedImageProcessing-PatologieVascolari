from pystackreg import StackReg
from skimage import io
import argparse
import numpy as np
from os import walk, sep
import cv2


def register(ref_path: str, mov_path: str) -> np.ndarray:
    ref = io.imread(ref_path)
    mov = io.imread(mov_path)

    sr = StackReg(StackReg.TRANSLATION)
    out_tra = sr.register_transform(ref, mov)

    return out_tra


def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('ref_dir',
                        type=str,
                        help='A required integer positional argument')

    parser.add_argument('target_dir',
                        type=str,
                        help='A required integer positional argument')

    parser.add_argument('dest_dir',
                        type=str,
                        help='A required integer positional argument')
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
        out_img = register(ref_list[i], target_list[i])
        cv2.imwrite(sep.join([args.dest_dir, f"REGISTERED_{i}.png"]), out_img)


if __name__ == "__main__":
    main()
