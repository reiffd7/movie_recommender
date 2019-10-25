import numpy as np

import matplotlib.pyplot as plt

def single_histogram(y_true, y_pred, bins=30, title=None, filepath=None):
    # Ensure that these are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Find residuals
    residuals = y_pred - y_true

    # Plot and either save to file or show
    _ = plt.hist(residuals, bins=bins)
    if title:
        _ = plt.title(title)
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()

def double_histogram(y1_true, y1_pred, y2_true, y2_pred, label1=None, label2=None, bins=30, title=None, filepath=None):
    # Ensure that these are NumPy arrays
    y1_true = np.array(y1_true)
    y1_pred = np.array(y1_pred)
    y2_true = np.array(y2_true)
    y2_pred = np.array(y2_pred)

    # Find residuals
    residuals1 = y1_pred - y1_true
    residuals2 = y2_pred - y2_true

    # Plot and either save to file or show
    _ = plt.hist(residuals1, bins=bins, label=label1, alpha=0.5)
    _ = plt.hist(residuals2, bins=bins, label=label2, alpha=0.5)
    _ = plt.legend()
    if title:
        _ = plt.title(title)
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


