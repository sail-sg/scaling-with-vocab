'''
The code to fit the tokenization function f(V)=D/H, 
where D is the number of tokens and H is the number of raw characters.
The fomula for f(V): f(V) = alog(V)**2+blog(V)+c
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pdb
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error, r2_score
from utils import relative_mse


def quadratic_log_fit(V, a,b,c):
    logv = np.log(V)
    return H*(a*(logv)**2 + b*(logv) + c)

def map_from_token_to_characters(V, D, H):
    '''
    The fitting process of the function f(V)=D/H
    '''
    params_quadraticlog, _ = curve_fit(quadratic_log_fit, V, D)

    print('The fitted (a,b,c) in f(V)=alog(V)**2+blog(V)+c is ', params_quadraticlog)
    print('max V is', math.exp(-params_quadraticlog[1]/(2*params_quadraticlog[0])))

if __name__ == '__main__':
    # Example data
    V = np.array([1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
                12288, 16384, 20480, 24576, 28672, 32768, 48128, 64512, 78848,
                96256, 128000, 256000, 512000, 1024000])
    D =  np.array([0.09915, 0.08441, 0.07771, 0.07380, 0.07107, 0.06894, 0.06725, 0.06593, 0.06484, 0.06387,
                0.06243, 0.06034, 0.05899, 0.05801, 0.05728, 0.05668, 0.05529, 0.05447, 0.05400, 0.05364,
                0.05321, 0.05242, 0.05172, 0.05140])
    H = 0.22326    
    map_from_token_to_characters(V, D, H)