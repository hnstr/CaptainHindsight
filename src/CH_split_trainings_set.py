import numpy as np
import json

path = '../../Data/'
filename = 'merged_train'

def open_data_file(filename):
    image_features = np.load(path + filename + '.npy')
    with open(path + filename + '.json') as f:
        image_captions = json.load(f)

    return image_features, image_captions

if __name__ == '__main__':
	image_features, image_captions = open_data_file(filename)

	chunck_size = 200
	image_features_chunks = [image_features[x:x+chunck_size] for x in range(0, len(image_features), chunck_size)]
	image_captions_chunks = [image_captions[x:x+chunck_size] for x in range(0, len(image_captions), chunck_size)]

	for i in range(len(image_features_chunks)):
		np.save(path + filename + str(i), image_features_chunks[i])

		with open(path + filename + str(i) + '.json', 'w') as f:
			json.dump(image_captions_chunks[i], f)