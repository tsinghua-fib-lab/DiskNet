import numpy as np


def MAE(y_true, y_pred, keep_step=False, keep_node=False):
    if not keep_step and not keep_node:
        return np.mean(np.abs(y_true - y_pred))
    elif keep_step and not keep_node:
        return np.mean(np.abs(y_true - y_pred), axis=(0,2,3))
    elif not keep_step and keep_node:
        return np.mean(np.abs(y_true - y_pred), axis=(0,1,3))
    
def MSE(y_true, y_pred, keep_step=False, keep_node=False):
    if not keep_step and not keep_node:
        return np.mean(np.square(y_true - y_pred))
    elif keep_step and not keep_node:
        return np.mean(np.square(y_true - y_pred), axis=(0,2,3))
    elif not keep_step and keep_node:
        return np.mean(np.abs(y_true - y_pred), axis=(0,1,3))

def RMSE(y_true, y_pred, keep_step=False):
    if not keep_step:
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
    else:
        return np.sqrt(np.mean(np.square(y_true - y_pred), axis=(0,2,3)))
    
def NMSE(y_true, y_pred, keep_step=False):
    if not keep_step:
        return np.mean(np.square(y_true - y_pred)) / (np.mean(np.square(y_true)) + 1e-7)
    else:
        return np.mean(np.square(y_true - y_pred), axis=(0,2,3)) / (np.mean(np.square(y_true), axis=(0,2,3)) + 1e-7)