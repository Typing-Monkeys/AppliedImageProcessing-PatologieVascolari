import cv2
from matplotlib import pyplot as plt
import numpy as np
from os import walk, sep


def process(fpath: str, area_low=300, verbose=False):
    img = cv2.imread(fpath)

    if verbose:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title('Original Img')

    # in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sogliatura e binarizzazione
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

    if verbose:
        plt.figure()
        plt.imshow(thresh, cmap='gray')
        plt.title('Thresholded Img')

    # individuo le componenti connesse 
    totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(thresh)

    output = np.zeros(thresh.shape, dtype="uint8")
    # Loop through each component
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]  
        
        if area_low < area:
            # Labels stores all the IDs of the components on the each pixel
            # It has the same dimension as the threshold
            # So we'll check the component
            # then convert it to 255 value to mark it white
            componentMask = (label_ids == i).astype("uint8") * 255
            
            # Creating the Final output mask
            output = cv2.bitwise_or(output, componentMask)
            #output = cv2.circle(output, tuple(centroid[i].astype(int)), radius=0, color=(0, 0, 255), thickness=2)
    
    if verbose:
        plt.figure()
        plt.imshow(output, cmap='gray')
        plt.title('Output Img')

    return output

def main(args: list[str]):
    source_dir = args[0]
    dest_dir = args[1]
    area = int(args[2]) if 2 < len(args) else 300

    for (dirpath, dirnames, filenames) in walk(source_dir):
        for filename in filenames:
            if filename.endswith('.png'): 
                img_full_path = sep.join([dirpath, filename]) 
                out_img = process(img_full_path, area_low=area)

                cv2.imwrite(sep.join([dest_dir, f"MASK_{filename}"]), out_img)
    

if __name__ == "__main__":
    from sys import argv

    main(argv[1:])