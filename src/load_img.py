import tensorflow as tf
from os import listdir
from os.path import isdir
def load_img(directory):
    facesTest = []
    facesTrain = []
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        for filename in listdir(path):
            if filename.find('.jpg') != -1:
                img_path = path + filename
                if filename.find('_0ID.jpg') != -1:
                    facesTrain.append(img_path)
                else:
                    facesTest.append(img_path)

    return facesTrain, facesTest

facesTrain, facesTest = load_img('E:/Hoc/BMI/venv/kieu/Data/')

print(facesTrain)

print(facesTest)
print("train" + str(len(facesTrain)))
print("test" + str(len(facesTest)))