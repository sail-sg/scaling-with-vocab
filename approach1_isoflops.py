'''
Select the data points that reach the lowest unigram-normalized loss under a certain Flops budget,
then we fit the relationships between (Nnv, Nv, H) with the Flops, respectively.

Fix the power for (Nnv, H) as 0.5 following Deepmind.

Fit the Nnv = np.exp(best_K1)*Flop_solutions**0.5
Fit the Nv = np.exp(best_K2)*Flop_solutions**alpha2
Fit the H = np.exp(best_K2)*Flop_solutions**0.5

Constraint: alpha1=beta=0.5

'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import huber
import pdb
import math
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.cm as cm
from utils import (D_to_H, relative_mse, Nnv_to_d, func_flops,
    interpolate, merge_nearest_flops, remove_outlier)



np.random.seed(42)  
a,b,c = 0.00639222, -0.15811069, 1.20470122

# # collect the exp data.
# ignore_folder = [
#         # 'V0004096', # if we need to omit partial data.
#     ]
# for exp in tqdm(sorted(ckpt_dir.glob(f'tiny_LLaMA_*IsoFLOP'))):
#     step_cnt = 0
#     num_ckpt_recode = len(list(exp.glob('*ckpt.txt')))
#     if any([i in exp.name for i in ignore_folder]):
#         continue
#     cur_flops, cur_loglossu,cur_lossu = [],[],[]
#     for idx,step_recode in enumerate(sorted(exp.glob('*ckpt.txt'))):
#         if not step_recode.is_file():
#             continue
#         step = float(step_recode.name.split('-')[1])

#         step_cnt += 1
#         model_size_name = exp.name.split('-')[0].split('_')[-1].lstrip('0')
#         d = embed_dim_dict[model_size_name]
#         N = model_size_dict[model_size_name]
#         V = float(exp.name.split('-')[1].split('_')[0][1:].replace('IsoFLOP',''))

#         steps_for_1epoch = steps_for_1epoch_dict[model_size_name]
#         D = max_D_dict[model_size_name] * (step/steps_for_1epoch) 
#         H = D_to_H(D,V)
#         Nnv = N - 2*16384*d
#         flops = 6*(Nnv + V*d)*D

#         points.append([Nnv, H, V])
#         with open(step_recode) as f:
#             loss_u = math.log(float(f.read()))
#         if all_samples is None:
#             all_samples = np.expand_dims(np.array([V, d, H, Nnv, flops, loss_u]), axis=0)
#         else:
#             all_samples = np.concatenate([all_samples, np.expand_dims(np.array([V, d, H, Nnv, flops,  loss_u]), axis=0)], axis=0)
    
#         cur_flops.append(flops)
#         cur_lossu.append(loss_u)
                      
#     print(exp.name, step_cnt)

df = pd.read_csv('exp_data.csv')
V_data = df['vocab_size']
d_data = df['embed_dim']
H_data = df['num_characters']
Nnv_data = df['Non_vocab_parameters']
flops_data = df['FLOPs']
lossu_data = df['Lossu']

num_model, num_v, num_eval = 6,10,20

L_values = np.array(lossu_data) # use lossu as the evaluation metric

length_bin = 1 
use_interpolation = True 
if use_interpolation:
    interpolated_Nnv, interpolated_H, interpolated_V, interpolated_flops, interpolated_loss = \
        interpolate(Nnv_data,H_data, V_data, flops_data, L_values, num_model, num_v, num_eval )

    V_data = np.concatenate([V_data, interpolated_V])
    H_data = np.concatenate([H_data, interpolated_H])
    Nnv_data = np.concatenate([Nnv_data, interpolated_Nnv])
    flops_data = np.concatenate([flops_data, interpolated_flops])
    L_values = np.concatenate([L_values, interpolated_loss])

    sortidx = np.argsort(flops_data)
    flops_data = np.sort(flops_data)
    V_data,H_data,Nnv_data = V_data[sortidx],H_data[sortidx],Nnv_data[sortidx]
    L_values = L_values[sortidx]

## select the point reaches the lowest loss under the same Flops budget
flops_idxs = np.argsort(flops_data)
flops_data = flops_data[flops_idxs]

V_data = V_data[flops_idxs]
H_data = H_data[flops_idxs]
Nnv_data = Nnv_data[flops_idxs]
L_values = L_values[flops_idxs]

considered_points = len(flops_data)

pivot = flops_data[0]
selected_ids = []
pivot = None
for i in range(0, considered_points, length_bin):
    offset = i
    min_id = np.argsort(L_values[i:i+length_bin])[0]
    if pivot == None or L_values[offset+min_id] < pivot:
        pivot = L_values[offset+min_id]
        selected_ids.append(offset+min_id)        
    

log_lossu_opt = L_values[selected_ids]
flopsopt = flops_data[selected_ids]
Vopt = V_data[selected_ids]
Hopt = H_data[selected_ids]
Nnvopt = Nnv_data[selected_ids]
assert len(flopsopt) == len(Vopt) == len(Hopt) == len(Nnvopt)

Nvopt = []
for each_Vopt, each_Nnvopt in zip(Vopt, Nnvopt):
    each_d = Nnv_to_d(each_Nnvopt)
    Nvopt.append(each_Vopt * each_d)



def LSE_Nnv_H(params, F):
    K = params
    return   K + 0.5*np.log(F)

def LSE(params, F):
    # Vopt = k*F^alpha
    # log(Vopt) = log(k)+alpha*log(F) = K+alpha*log(F)
    # fit K, alpha, where K = log(k)
    K, alpha = params
    return   K + alpha*np.log(F)

def objective_function_Nvopt(params, delta=0.001):
    prediction = LSE(params, flopsopt)
    residuals = (prediction - np.log(Nvopt))
    return np.sum(huber(delta, residuals))

def objective_function_Hopt(params, delta=0.001):
    prediction = LSE_Nnv_H(params, flopsopt)
    residuals = (prediction - np.log(Hopt))
    return np.sum(huber(delta, residuals))

def objective_function_Nnvopt(params, delta=0.001):
    prediction = LSE_Nnv_H(params, flopsopt)
    residuals = (prediction - np.log(Nnvopt))
    return np.sum(huber(delta, residuals))


best_alpha_set=[]
best_K_set = []
best_mse_set, best_r2_set = [],[]
fit = True
if fit:
    print('start fitting...')
    for time in range(3):
        best_mse = float('inf')
        best_r2 = 0
        best_mse_init_guess, best_r2_init_guess = None, None
        best_mse_guess, best_r2_guess = None, None
        cnt = 0
        for init_K in  np.linspace(-20, 15, 20):
            for init_alpha in np.linspace(0, 1, 20):
                cnt += 1
                if cnt % 500 == 0:
                    print('The number of init guess: ',cnt)
                if time == 1:
                    initial_guess = [init_K, init_alpha]
                else:
                    initial_guess = [init_K]

                if time == 0:
                    result = minimize(objective_function_Nnvopt, initial_guess,  
                        method='L-BFGS-B',)
                    data_actual = np.log(Nnvopt)
                    data_predicted = np.array([LSE_Nnv_H(result.x, F) for F in flopsopt])
                elif time == 1:
                    result = minimize(objective_function_Nvopt, initial_guess,  
                        method='L-BFGS-B', )
                    data_actual = np.log(Nvopt)
                    data_predicted = np.array([LSE(result.x, F) for F in flopsopt])
                elif time == 2:
                    result = minimize(objective_function_Hopt, initial_guess,  
                        method='L-BFGS-B', )
                    data_actual = np.log(Hopt)
                    data_predicted = np.array([LSE_Nnv_H(result.x, F) for F in flopsopt])

                                        
                mse = relative_mse(data_actual, data_predicted)
                r2 = r2_score(data_actual, data_predicted)

                if mse < best_mse:
                    best_mse = mse
                    best_mse_init_guess = initial_guess
                    best_mse_guess = result.x
                if r2 > best_r2:
                    best_r2 = r2
                    best_r2_init_guess = initial_guess
                    best_r2_guess = result.x

        print(f"MSE (good MSE near to 0): {best_mse}\n\
            best_r2 (good r2 near to 1): {best_r2}\n\
            best_mse_init_guess is {best_mse_init_guess}\n\
            best_mse_guess is {best_mse_guess}\n\
            "
        )
        if time == 1:
            best_K, best_alpha = best_mse_guess
        else:
            best_K = best_mse_guess[0]
            best_alpha = 0.5
            
        best_K_set.append(best_K)
        best_alpha_set.append(best_alpha)   
        best_mse_set.append(best_mse)     
        best_r2_set.append(best_r2)

def Nnvopt_to_flops(Nnv):
    return  (Nnv/np.exp(best_K_set[0])) ** (1/0.5)

def Nnvopt_to_Nvopt(Nnv):
    F = ( Nnv/np.exp(best_K_set[0])) ** (1/0.5)
    return np.exp(best_K_set[1])* F ** best_alpha_set[1]

def Flops_to_Nvopt(F):
    return np.exp(best_K_set[1])* F ** best_alpha_set[1]

for test_Nnv in [2.87*10**9, 3*10**9, 7*10**9, 13*10**9, 30*10**9, 70*10**9, 130*10**9, 300*10**9]:
    d = Nnv_to_d(test_Nnv)
    flops = Nnvopt_to_flops(test_Nnv)
    V = int(Nnvopt_to_Nvopt(test_Nnv)/d)
    Nv = V*d
    print(f'Approach1: Nnv={test_Nnv:.1e}, FLOPs={flops:.1e}, Vopt-isoflops:{V}, Nv-isoflops:{Nv/10**9}B')
