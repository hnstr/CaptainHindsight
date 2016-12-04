import json
import numpy as np
import os.path
import pickle

def open_data_file(filename):
    path = '../../Data/'

    image_features = np.load(path + filename + '.npy')
    with open(path + filename + '.json') as f:
        image_captions = json.load(f)

    return image_features, image_captions

# If you get a keyerror, it probably is because you changed the dataset you train on,
# while you still use the old mapping. Remove mapping.p to resolve
def get_word_int_mappings(filename):
    if os.path.isfile('mappings.p'):
        with open('mappings.p', 'rb') as f:
            (word2int, int2word) = pickle.load(f)
            return word2int, int2word

    else:
        _, image_captions = open_data_file(filename)
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

        pickle.dump((word2int, int2word), open('mappings.p', 'wb'))

    return word2int, int2word