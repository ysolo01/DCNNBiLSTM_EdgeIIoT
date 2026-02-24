import numpy as np
import pandas as pd
import os

np.random.seed(42)

num_samples = 20000
num_features = 76

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 3, num_samples)

data = pd.DataFrame(X)
data["label"] = y

os.makedirs("../data", exist_ok=True)
data.to_csv("../data/dataset.csv", index=False)

print("Dataset generated successfully.")