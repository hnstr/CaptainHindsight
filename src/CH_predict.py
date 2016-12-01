
def predict_caption(model, image_feature):

    # Preallocate empty array
    predictions = []

    for i in range(len(image_features)):
        feature = image_features[i, :]
        # Predict new output based on single image feature vector
        # TODO: update batch size
        prediction = predict(model, feature, batch_size=32, verbose=0)

        predictions.append(prediction)

    return predictions
