import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import MinMaxScaler
import os

SEED = 42

def load_and_preprocess(percent10=False):

    print("Downloading KDDCup'99 dataset (this may take a minute)...")


    data = fetch_kddcup99(percent10=percent10, shuffle=True, random_state=SEED, download_if_missing=True)
    X_raw = data.data
    y_raw = data.target

    df = pd.DataFrame(X_raw)

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("bytes"):
            # decode bytes if needed
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x))
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    feature_names = [f"f{i+1}" for i in range(df.shape[1])]

    y = np.array([0 if (isinstance(t, (bytes, bytearray)) and t.decode('utf-8').startswith('normal')) or (str(t).startswith('normal')) else 1 for t in y_raw])

    X = df.values.astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names

if __name__ == '__main__':
    X, y, names = load_and_preprocess()
    print('X shape:', X.shape)
    print('y distribution:', pd.Series(y).value_counts().to_dict())
