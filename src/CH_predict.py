import numpy as np


def predict_caption(model, image_feature, word2int, int2word):
    predicted_sentence = np.array([[word2int['<s>']]])
    image_feature = np.array([image_feature])

    end_of_sentence = False
    sentence_length = 0

    while not end_of_sentence:
        prediction = model.predict([image_feature, predicted_sentence], verbose=0)

        # pick most likely word to occur
        encoded_word = np.argmax(prediction)
        predicted_sentence = np.append(predicted_sentence[0], encoded_word)
        predicted_sentence = np.array([predicted_sentence])

        # naive sampling of sentence
        if (encoded_word == word2int['</s>']):
            end_of_sentence = True
        elif (sentence_length > 60):
            end_of_sentence = True
        else:
            sentence_length += 1

    return predicted_sentence[0]
