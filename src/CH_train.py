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

    for image_caption_pair in image_captions:
        for caption in image_caption_pair[1]:
            for token in caption:
                # TODO: Do we normalize?
                word_types.add(token)

    word_types.add('<s>')
    word_types.add('</s>')

    sorted_word_types = sorted(list(word_types))
    word2int = dict((c, i) for i, c in enumerate(sorted_word_types))
    int2word = dict((i, c) for i, c in enumerate(sorted_word_types))

    return word2int, int2word

def get_data_pairs(image_features, image_captions, word2int):
    # TODO: make pad_lenght equal to max caption length
    pad_length = 60
    x = np.zeros((len(image_features), len(image_features[0]) + pad_length))
    t = np.zeros((len(image_features), pad_length))

    for counter in range(len(image_features)):

        feature = image_features[counter, :]

        # Iterate over 4 possible captions, convert them to int and place in IO arrays
        for i in range(1, 5):
            caption = image_captions[counter][1][i]
            x_vec = np.zeros(pad_length)  # initialize empty word vector
            t_vec = np.zeros(pad_length)

            word_count = 0

            # Add begin of sentence markers
            x_vec[word_count] = word2int['<s>']
            t_vec[word_count] = word2int['<s>']

            # Process sentence
            for word in caption:
            	word_count += 1
                word_int = word2int[word]
                x_vec[word_count] = word_int
                t_vec[word_count] = word_int

            # Add end of sentence markers
            x_vec[word_count+1] = word2int['</s>']
            t_vec[word_count+1] = word2int['</s>']
                

            x_ = np.hstack((feature, x_vec))
            # TODO: is this necessary?
            x_ = x_ / float(len(word2int))

            x[counter, :] = x_
            t[counter, :] = t_vec

            y = np_utils.to_categorical(t)

    return x, y

def train(model, x, y):
	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# TODO: Add batchsize
	model.fit(X, y, nb_epoch=1, callbacks=callbacks_list)


if __name__ == '__main__':
	# Retrieve image features and captions from files
    image_features, image_captions = open_data_file('merged_val')
    # Get word-int mappings
    word2int, int2word = get_word_int_mappings(image_captions)
    # Convert to approriate input-output pairs
    x, y = get_data_pairs(image_features, image_captions, word2int)
    # Build model
    model = get_LSTM_model(len(image_features[0]), len(word2int))
    # Train the model
    train(model, x, y)
