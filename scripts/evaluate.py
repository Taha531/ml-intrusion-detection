import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODELS_DIR = os.path.join(os.getcwd(), 'models')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')


def plot_cm(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.savefig(outpath)
    plt.close()


if __name__ == '__main__':
    from scripts.preprocess import load_and_preprocess
    from sklearn.model_selection import train_test_split

    X, y, _ = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.joblib'):
            model = joblib.load(os.path.join(MODELS_DIR, fname))
            y_pred = model.predict(X_test)
            outpath = os.path.join(RESULTS_DIR, f'cm_{fname.replace(".joblib","")}.png')
            plot_cm(y_test, y_pred, f'Confusion matrix — {fname}', outpath)
            print('Saved', outpath)

    print('Evaluation done. Confusion matrices are in results/confusion_matrices or results/')