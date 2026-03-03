import numpy as np

def predict_with_uncertainty(model, X_input, n_iter=100):
    predictions = []
    for _ in range(n_iter):
        predictions.append(model(X_input, training=True))

    predictions = np.array(predictions)

    prediction_mean = np.mean(predictions, axis=0)
    prediction_std = np.std(predictions, axis=0)

    return prediction_mean, prediction_std