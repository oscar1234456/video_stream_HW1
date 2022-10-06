import numpy as np

def mape(actual, pred):
    actual, pred = np.array(actual), np.squeeze(np.array(pred))
    # print(f"actual: {actual}")
    # print(f"pred:{pred.shape}")
    return np.mean(np.abs((actual - pred) / actual)) * 100

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.squeeze(np.array(predictions))
    return np.mean(np.abs(y_true - predictions))


if __name__ == "__main__":
    print(mape([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24], [[37], [40], [46], [44], [46], [50], [45], [44], [34], [30], [22], [23]]))
    print(mae([12, 13, 14, 15, 15, 22, 27], [[11], [13], [14], [14], [15], [16], [18]]))