import json
import numpy as np

from CH_model import get_LSTM_model

def open_data_file(filename):
	path = '../../Data/'

	image_features = np.load(path + filename + '.npy')
	with open(path + filename + '.json') as f:
		image_captions = json.load(f)

	return image_features, image_captions

def get_word_int_mappings(image_captions):
	word_types = set()
	i = 0
	for image_caption_pair in image_captions:
		for caption in image_caption_pair[1]:
			for token in caption:
				# TODO: Do we normalize?
				word_types.add(token)

	sorted_word_types = sorted(list(word_types))
	word2int = dict((c, i) for i, c in enumerate(sorted_word_types))
	int2word = dict((i, c) for i, c in enumerate(sorted_word_types))

	return word2int, int2word

# TODO: Get data pairs:
# x = [I, S0, ..., SN-1]
# t = [S0, ...,, SN]
# Write to file to save time next run
def get_data_pairs(filename):
	pass

# TODO: Train model
# Use ModelCheckpoint to write weight to file after every epocg
def train(model, x, y):
	pass

if __name__ == '__main__':
	image_features, image_captions = open_data_file('merged_val')
	word2int, int2word = get_word_int_mappings(image_captions)

	model = get_LSTM_model(4096, 26)
