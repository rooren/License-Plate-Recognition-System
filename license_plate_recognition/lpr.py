from cca import get_license_plate_box 
from character_segmentation import character_segmentation
import os
from character_recognition import read_training_data, cross_validation, cross_val_score
from sklearn.svm import SVC
import argparse

def main():
    parser = argparse.ArgumentParser(
                    prog='License-Plate-Recognition-System',
                    description='License Plate Recognition System using skimage and ML',)
    parser.add_argument('-f', '--file')
    # using the given image 
    plate_like_objects = get_license_plate_box(parser.parse_args().file)
    segmentation_characters, segmentation_col_list = character_segmentation(plate_like_objects)
    current_dir = os.path.dirname(os.path.realpath(__file__))

    training_dataset_dir = os.path.join(current_dir, 'training_directory')

    image_data, target_data = read_training_data(training_dataset_dir)

    # the kernel can be 'linear', 'poly' or 'rbf'
    # the probability was set to True so as to show
    # how sure the model is of it's prediction
    svc_model = SVC(kernel='linear', probability=True)

    cross_validation(svc_model, 4, image_data, target_data)

    # let's train the model with all the input data
    svc_model.fit(image_data, target_data)

    classification_result = []
    for each_character in segmentation_characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = svc_model.predict(each_character)
        classification_result.append(result)

    print(classification_result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    column_list_copy = segmentation_col_list[:]
    segmentation_col_list.sort()
    rightplate_string = ''
    for each in segmentation_col_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    print(rightplate_string)

if __name__ == "__main__":
    main()