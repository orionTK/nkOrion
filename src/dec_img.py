# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir

import cv2
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray

def extract_img(filename):
    img = cv2.imread(filename)  # đọc ra ảnh BGR
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert lại thành ảnh RGB
    face_array = asarray(img)
    return face_array


def load_faces(directory,i):
	facesTest , facesTrain = [], []
	# dem=0;
	k = 0;
	# enumerate files
	for filename in listdir(directory):
		# path

		if filename.find('.jpg') != -1:
			path = directory + filename
			face = extract_img(path)
			if filename.find('_0ID.jpg') != -1:
				facesTrain.append(face)
			else:
				facesTest.append(face)

	if i == 0:
		return facesTrain
	else:
		return facesTest


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, i):
    x = []
    y = []
	# enumerate folders, on per class
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        #load
        faces= load_faces(path, i);
        # create labels
        labels = [subdir for _ in range(len(faces))]
        #summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)

# load train dataset
trainx, trainy = load_dataset('E:/Hoc/BMI/venv/kieu/Data/', 0)

# # load test dataset
testx, testy = load_dataset('E:/Hoc/BMI/venv/kieu/Data/', 1)
print("train")
print(trainx.shape, trainy.shape)
print("test")
print(testx.shape, testy.shape)

# # save arrays to one file in compressed format
savez_compressed('dataf.npz', trainx, trainy, testx, testy)


