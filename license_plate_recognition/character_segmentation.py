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
    license_plate = np.invert(get_license_plate_object(plate_like_objects))

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
    return (characters, column_list)

def get_license_plate_object(plate_like_objects):
    if not plate_like_objects:
        raise LPRError("License plate could not be located")

    if len(plate_like_objects) == 1:
        return plate_like_objects[0]
    else:
        license_plate = validate_plate(plate_like_objects)
    return license_plate


def validate_plate(candidates):
    """
    validates the candidate plate objects by using the idea
    of vertical projection to calculate the sum of pixels across
    each column and then find the average.

    This method still needs improvement

    Parameters:
    ------------
    candidate: 3D Array containing 2D arrays of objects that looks
    like license plate

    Returns:
    --------
    a 2D array of the likely license plate region

    """
    lowest_average = -1
    license_plate = []
    i = -1
    for candidate in candidates:
        i = i +1
        height, width = candidate.shape
        threshold_value = threshold_otsu(candidate) 
        thresh_candidate = candidate < threshold_value
        total_white_pixels = 0
        for column in range(width):
            total_white_pixels += sum(thresh_candidate[:, column])
        
        average = float(total_white_pixels) / width
        if average <= lowest_average or lowest_average == -1:
            lowest_average = average
            license_plate = candidate

    return license_plate



class LPRError(Exception):
    """
    custom error for LPR class 
    """
    pass