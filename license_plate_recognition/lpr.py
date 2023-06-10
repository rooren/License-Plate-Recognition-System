from cca import get_license_plate_box 
from character_segmentation import character_segmentation
import os
from character_recognition import read_training_data, cross_validation, cross_val_score
# from sklearn.externals import joblib
from sklearn.svm import SVC

def main():

    plate_like_objects = get_license_plate_box("test_imgs/car5.jpg")
    a = character_segmentation(plate_like_objects)
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

    # we will use the joblib module to persist the model
    # into files. This means that the next time we need to
    # predict, we don't need to train the model again
    #save_directory = os.path.join(current_dir, 'models/svc/')
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # joblib.dump(svc_model, save_directory+'/svc.pkl')

if __name__ == "__main__":
    main()