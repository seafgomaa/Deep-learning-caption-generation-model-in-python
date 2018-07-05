from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

arr=[]
features = dict()
#Put images pathes in array arr[]
def x(i):
    for n in listdir(i):
         arr.append(i + '/' + n)
         
# extract features from each photo in the directory
def extract_features():
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract features from each photo

	
	for filename in arr:
		# load an image from file		
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = (filename.split('/')[1]).split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % filename)
	return features

# extract features from all images
x('Flicker8k_Dataset')
features = extract_features()
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
