from sklearn.metrics import classification_report
import numpy as np

def train_model(model, X_train, y_train, X_test, y_test):

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )

    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred))

    model.save("../dcnn_bilstm_model.h5")
    print("Model saved.")