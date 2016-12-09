import numpy as np


def predict_caption(model, image_feature, word2int, int2word):

    start_token = np.ones((np.size(image_feature), 1)) * word2int['<s>']

    previous_token = start_token
    end_of_sentence = False
    sentence_length = 0

    predicted_sentence = []


    while not end_of_sentence:
        prediction = model.predict([image_feature, previous_token], batch_size=32, verbose=1)

        encoded_word = np.argmax(prediction)

        # Naive sampling of sentence
        if (encoded_word == word2int['</s>']):
            end_of_sentence = True
            print('End of sentence marker encountered')
        elif (sentence_length > 60):
            end_of_sentence = True
            print('Max sentence length reached')
        else:
            sentence_length += 1
            # print(encoded_word)
            print(int2word[encoded_word])
            previous_token = np.ones((np.size(image_feature), 1)) * encoded_word
            predicted_sentence.append(encoded_word)

    # predictions.append(prediction)

    return predicted_sentence
