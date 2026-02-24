import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(path):

    data = pd.read_csv(path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # reshape for CNN + LSTM
    X_train = X_train.reshape(X_train.shape[0], 76, 1)
    X_test = X_test.reshape(X_test.shape[0], 76, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test