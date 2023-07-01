# License-Plate-Recognition-System
License Plate Recognition System using skimage and ML

First the progrem will scan for the possible locations of the licesnse plate 
![image](https://github.com/rooren/License-Plate-Recognition-System/assets/58310236/1921ef1d-f6d0-4758-a4cd-45b7ac05fea6)

After locating the correct rectangle of the license plate it will attempt to separate the singel characters
![image](https://github.com/rooren/License-Plate-Recognition-System/assets/58310236/bb8f0217-5dfc-43a9-8d50-230823486c1a)

Using the SVC ML model we had trained it will attempt to identify each letter

output : ![image](https://github.com/rooren/License-Plate-Recognition-System/assets/58310236/de40bb7e-e2d6-4789-94f7-9f33abf5e132)


## **Dependencies**
The program was written with python 2.7 and the following python packages are required
* [Numpy](http://docs.scipy.org/doc/numpy-1.10.0) Numpy is a python package that helps in handling n-dimensional arrays and matrices
* [Scipy](http://scipy.org) Scipy for scientific python
* [Scikit-image](http://scikit-image.org/) Scikit-image is a package for image processing
* [Scikit-learn](http://scikit-learn.org/) Scikit-learn is for all machine learning operations
* [Matplotlib](http://matplotlib.org) Matplotlib is a 2D plotting library for python

## **How to use**
1. Clone the repository or download the zip 
2. Change to the cloned directory (or extracted directory)
3. Create a virtual environment with virtualenv
4. Install all the necessary dependencies by using pip `pip install -r requirements.txt`
5. Start the program `python lpr.py -f <image_path>`
