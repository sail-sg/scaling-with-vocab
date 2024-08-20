'''
Use all the data points collected during the experiments.

We design the loss formula as:
lossu(Nnv,V,H) = -E + A1/[Nnv]^alpha1 + A2/[Vd]^alpha2 + B/[F/(6(Nv+Vd))]^beta
lossu is negative typically, so I set the constant -E instead of E 

Then, we use the fitted (E, A1, alpha1, A2, alpha2, B, beta) to predict the optimal vocabulary size
by computing the solution of dlossu/dV=0.

Constraint: alpha1=beta,  low < alpha1 < 1,  low < beta < 1
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import huber
import pdb
import math
from scipy.optimize import fsolve
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from sklearn.metrics import r2_score
import matplotlib.cm as cm
import warnings
import contextlib
from utils import (D_to_H, H_to_D, interpolate, func_flops, Nnv_to_d, 
    merge_nearest_flops, relative_mse)

np.random.seed(42)  # For reproducibility

def dl_dv(V, Nnv, d, F):
    term1 = 0  # Derivative of -E
    term2 = 0  # Derivative of A1/[Nnv]^alpha1
    term3 = -alpha2 * A2 * d / (V * d) ** (alpha2 + 1)
    u = F / (6 * (Nnv + V * d))
    du_dV = F * d / (6 * (Nnv + V * d) ** 2)
    term4 = beta * B * du_dV / (u ** (beta + 1))
    
    return term1 + term2 + term3 + term4


df = pd.read_csv('exp_data.csv')

V_data = df['vocab_size']
d_data = df['embed_dim']
H_data = df['num_characters']
Nnv_data = df['Non_vocab_parameters']
flops_data = df['FLOPs']
lossu_data = df['Lossu']

# We follow DeepMind Chinchilla's approach to omit the points with flops < 10e18
flops_idxs = np.argsort(flops_data)
flops_data = flops_data[flops_idxs]
num_points = np.sum(flops_data < 2.18 * 10 ** 17)

V_data = V_data[flops_idxs][num_points:]
H_data = H_data[flops_idxs][num_points:]
Nnv_data = Nnv_data[flops_idxs][num_points:]
d_data = d_data[flops_idxs][num_points:]
lossu_data = lossu_data[flops_idxs][num_points:]

## normalization
V_data = V_data / 1_000
H_data = H_data / 1_000_000_000
Nnv_data = Nnv_data / 1_000_000
d_data = d_data / 1000
print('Num of points for fitting: ', len(lossu_data))


def LSE(params, Nnv, V, d, H):
    # fit a, b, e, alpha, beta, where A = exp(a), B = exp(b), E = exp(e)
    a1, a2, b, e, alpha2, beta = params
    alpha1 = beta
    term1_1 = a1 - alpha1 * np.log(Nnv)
    term1_2 = a2 - alpha2 * np.log(V*d)
    term2 = b - beta * np.log(H_to_D(H,V))
    term3 = e
    return (np.exp(term1_1) + np.exp(term1_2) + np.exp(term2) - np.exp(term3))


def objective_function(params, delta=0.001):
    predict_lossu = LSE(params, Nnv_data, V_data, d_data, H_data)
    residuals = (predict_lossu - lossu_data)
    return np.sum(huber(delta, residuals))



'''
Given the (Nnv, d, H), we predict the optimal V based on the fitted loss function
'''
print('start fitting...')
low_bound = 0.1
best_mse = float('inf')
best_r2 = 0
best_mse_init_guess, best_r2_init_guess = None, None
best_mse_guess, best_r2_guess = None, None
best_data_predicted = None
cnt = 0    
for init_a1 in  np.linspace(0, 5, 3):
    for init_a2 in  np.linspace(0, 5, 3):
        for init_b in np.linspace(0, 5, 3):
            for init_e in  np.linspace(0, 2, 2):
                for init_alpha2 in  np.linspace(0, 1, 3):
                    for init_beta in  np.linspace(0, 1, 3):
                        cnt += 1
                        if cnt % 100 == 0:
                            print('The number of init guess: ', cnt, 'best_mse', best_mse)
                        initial_guess = [init_a1, init_a2, init_b, init_e, init_alpha2, init_beta]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)                            
                            result = minimize(objective_function, initial_guess,
                                            method='L-BFGS-B',
                                            )
                        data_actual = lossu_data
                        data_predicted = np.array([LSE(result.x, Nnv,V,d,H) for Nnv,V,d,H in zip(Nnv_data, V_data, d_data, H_data)])
                        mse = relative_mse(data_actual, data_predicted)
                        r2 = r2_score(data_actual, data_predicted)

                        constraint = low_bound < result.x[4] < 1 and low_bound <  result.x[5] < 1

                        if mse < best_mse and constraint:
                            best_mse = mse
                            best_mse_init_guess = initial_guess
                            best_mse_guess = result.x
                            best_data_predicted = data_predicted
                        
                        if r2 > best_r2 and constraint:
                            best_r2 = r2
                            best_r2_init_guess = initial_guess
                            best_r2_guess = result.x

print(f"RMSE (good RMSE near to 0): {best_mse}\n\
best_r2 (good r2 near to 1): {best_r2}\n\
best_mse_init_guess is {best_mse_init_guess}\n\
best_mse_guess is {best_mse_guess}\n")       
            
a1,a2,b,e,alpha2,beta = best_mse_guess
alpha1 = beta
A1,A2,B,E = math.exp(a1),math.exp(a2),math.exp(b),math.exp(e)
print(f'A1={A1}, A2={A2}, B={B}, E={E},\nalpha1={alpha1}, alpha2={alpha2}, beta={beta}')


from approach1_isoflops import Nnvopt_to_flops
Nnvs =[2.87*10**9, 3*10**9, 7*10**9, 13*10**9, 30*10**9, 70*10**9, 130*10**9, 300*10**9]
Nnvs = np.array(Nnvs, dtype=np.float64)
Nvs = []
for Nnv in Nnvs:
    # normalization
    d = Nnv_to_d(Nnv)
    F = Nnvopt_to_flops(Nnv)
    Nnv = Nnv / 1_000_000
    d = d / 1_000   
    F = F / (1_000_000_000*1_000_000)
    v = fsolve(dl_dv, 1, args=(Nnv,d,F,))[0]
    
    # de-normalization
    d = d * 1000
    Nnv = Nnv * 1_000_000
    F = F * (1_000_000_000*1_000_000)
    v = int(v*1000)
    Nv = v * d
    Nvs.append(Nv)
    print(f'Approach3: Nnv={Nnv:.1e}, FLOPs={F:.1e}, Vopt-isoloss:{v}, Nv={Nv/10**9}B')


## Study the cases of under-training / over-training.
print('\nStudy the cases of under-training / over-training. Example Nnv=302M')
for time in [0.2,0.3,0.5,1,2,3,4,5,6]:
    # normalization
    Nnv = 302*10**6
    d = Nnv_to_d(Nnv)
    F = time * Nnvopt_to_flops(Nnv)
    Nnv = Nnv / 1_000_000
    d = d / 1_000   
    F = F / (1_000_000_000*1_000_000)
    v = fsolve(dl_dv, 1, args=(Nnv,d,F,))[0]   
    print(f'Approach3: Nnv={Nnv*1_000_000:.1e}, FLOPs={F*(1_000_000_000*1_000_000):.1e}, Vopt-isoloss:{int(v*1000)}')

print('\nThe scaling factor of Nv with respect to Nnv:')
for i in range(1, len(Nnvs)):
    dNnv = Nnvs[i] / Nnvs[0]
    dNv = Nvs[i] / Nvs[0]
    diff = math.log(dNv) / math.log(dNnv)
    print(f"Approach3: Nnv={Nnvs[i]:.1e}, Nv={Nvs[i]:.1e}, log(dNv)/log(dNnv)={diff:.4f}")
