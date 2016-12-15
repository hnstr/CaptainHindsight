import json
import numpy as np
import os.path
import gc
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

from CH_model import get_LSTM_model
from CH_mappings import get_word_int_mappings
import CH_predict

init_training = 1
filename_weights_to_import = 'weights/?.hdf5'
n_epochs = 100
filename_to_train_on = 'merged_train'
n_files_trainingset_is_splitted_in = 716

def open_data_file(filename):
	path = '../../Data/'

	image_features = np.load(path + filename + '.npy')
	with open(path + filename + '.json') as f:
		image_captions = json.load(f)

	return image_features, image_captions

def get_data_pairs(filename, word2int):
	image_features, image_captions = open_data_file(filename)

	x_img, x_lang, t = [], [], []

	for image_counter in range(len(image_features)):
		for caption_counter in range(5):
			caption = image_captions[image_counter][1][caption_counter]
			full_caption = ['<s>'] + caption + ['</s>']

			for word_counter in range(1,len(full_caption)):
				cap_to_int = [word2int[word] for word in full_caption[:word_counter+1]]

				x_img.append(image_features[image_counter])
				x_lang.append(np.array(cap_to_int[:-1]) / float(len(word2int)))
				t.append(cap_to_int[-1])


	x_img = np.array(x_img)
	x_lang = pad_sequences(x_lang)
	y = np_utils.to_categorical(t, len(word2int))

	return x_img, x_lang, y

def train(model, x_img, x_lang, y):
	filepath="weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	model.fit([x_img, x_lang], y, nb_epoch=1, batch_size=32, callbacks=callbacks_list)


if __name__ == '__main__':
	# Get word-int mappings
	word2int, int2word = get_word_int_mappings(filename_to_train_on)

	# Build model
	model = get_LSTM_model(4096, len(word2int), init_training, filename_weights_to_import)

	if not os.path.isfile('../../Data/' + filename_to_train_on + str(n_files_trainingset_is_splitted_in - 1) + '.npy') or \
		   os.path.isfile('../../Data/' + filename_to_train_on + str(n_files_trainingset_is_splitted_in) + '.npy'):
		print "Something went wrong!!"

	for epoch in range(n_epochs):
		for n in np.random.permutation(n_files_trainingset_is_splitted_in):
			gc.collect()

			full_filename_to_train_on = filename_to_train_on
			if n_files_trainingset_is_splitted_in > 1:
				full_filename_to_train_on += str(n)

			# Convert to approriate input-output pairs
			x_img, x_lang, y = get_data_pairs(full_filename_to_train_on, word2int)

			# Train the model
			train(model, x_img, x_lang, y)

	# Predict new outputs
	# image_features_val, image_captions_val = open_data_file('merged_val')
	# predictions = predict_caption(model, image_features_val)
