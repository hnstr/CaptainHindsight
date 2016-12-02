import json
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from CH_model import get_LSTM_model
import CH_predict

training_mode = 1
filename_weights_to_import = 'weights/?.hdf5'
n_epochs = 1

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
    x_img = []
    x_lang = np.zeros((len(image_features), pad_length))
    t = np.zeros((len(image_features), pad_length), dtype=int)

    for counter in range(len(image_features)):
        feature = image_features[counter, :]
        x_img.append(feature)

        # Iterate over 4 possible captions, convert them to int and place in IO arrays
        for i in range(1, 5):
            caption = image_captions[counter][1][i]
            x_vec = np.zeros(pad_length, dtype=int)  # initialize empty word vector
            t_vec = np.zeros(pad_length, dtype=int)

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

            # x_ = np.hstack((feature, x_vec))
            # TODO: is this necessary?
            x_ = x_vec / float(len(word2int))
            x_lang[counter, :] = x_
            t[counter, :] = t_vec

    x_img = np.array(x_img)
    y = np_utils.to_categorical(t)

    return x_img, x_lang, y

def train(model, x_img, x_lang, y):
    if training_mode:
    	# define the checkpoint
    	filepath="weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    	callbacks_list = [checkpoint]

    	# TODO: Add batchsize
    	model.fit([x_img, x_lang], y, nb_epoch=n_epochs, batch_size=32, callbacks=callbacks_list)

    else:
        # load the network weights
        model.load_weights(filename_weights_to_import)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

if __name__ == '__main__':
	# Retrieve image features and captions from files
    image_features, image_captions = open_data_file('merged_train')
    # Get word-int mappings
    word2int, int2word = get_word_int_mappings(image_captions)
    # Convert to approriate input-output pairs
    x_img, x_lang, y = get_data_pairs(image_features, image_captions, word2int)
    # Build model
    model = get_LSTM_model(len(image_features[0]), len(word2int))
    # Train the model
    train(model, x_img, x_lang, y)

    # Predict new outputs
    # image_features_val, image_captions_val = open_data_file('merged_val')
    # predictions = predict_caption(model, image_features_val)