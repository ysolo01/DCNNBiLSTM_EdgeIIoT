from generate_dataset import *
from preprocess import load_data
from model import build_model
from train import train_model

def main():

    # 1️⃣ generate dataset
    generate_dataset = True
    if generate_dataset:
        import generate_dataset

    # 2️⃣ preprocess
    X_train, X_test, y_train, y_test = load_data("../data/dataset.csv")

    # 3️⃣ build model
    model = build_model()

    # 4️⃣ train
    train_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()