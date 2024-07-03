import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def R_squared(y, y_pred):
    return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def MAPE(y, y_pred):
    return 100 * np.nanmean(np.abs(y - y_pred) / np.abs(y))
