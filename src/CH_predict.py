import numpy as np


def predict_caption(model, image_feature, word2int, int2word):

    start_token = np.ones((np.size(image_feature), 1)) * word2int['<s>']

    previous_token = start_token
    end_of_sentence = False
    sentence_length = 0

    predicted_sentence = [word2int['<s>']]

    while not end_of_sentence:
        prediction = model.predict([image_feature, previous_token], batch_size=32, verbose=0)

        # pick most likely word to occur
        encoded_word = np.argmax(prediction)
        predicted_sentence.append(encoded_word)

        # naive sampling of sentence
        if (encoded_word == word2int['</s>']):
            end_of_sentence = True
            # print('End of sentence marker encountered')
        elif (sentence_length > 60):
            end_of_sentence = True
            # print('Max sentence length reached')
        else:
            predicted_token = np.ones((np.size(image_feature), 1)) * encoded_word
            previous_token = np.hstack((previous_token, predicted_token))
            sentence_length += 1

    return predicted_sentence
