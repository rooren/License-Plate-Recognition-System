from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.transform import resize

def character_segmentation(plate_like_objects:list):
        # The invert was done so as to convert the black pixel to white pixel and vice versa
    license_plate = np.invert(plate_like_objects[1])
    labelled_plate = measure.label(license_plate)
    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")
    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some
    character_dimensions = (0.1*license_plate.shape[0], 0.95*license_plate.shape[0], 0.05*license_plate.shape[1], 0.4*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    counter=0
    column_list = []
    # print("min and max heigt " +str(max_height) +" "+str(min_height) +" ")
    # print("min and max width " +str(max_width) +" "+str(min_width) +" ")

    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0
        # print("-----------")
        # print("hight = " + str(region_height))
        # print("width = " + str(region_width))

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                        linewidth=2, fill=False)
            ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)
    plt.show()
    return characters
