'''
We provide the derivative-based approach to predict the optimal Vocabulary size by
either the FLops or the non-vocabulary parameters Nnv.

Use the derivative-based approach to predict the Nv given Nnv, then 
we fit gamma for dNv = dNnv **gamma, where dNv = Nv2/Nv1, dNnv = Nnv2/Nnv1.

Constraint: gamma < 1

Then, we compare the predictions between the derivative-based approach and IsoFLOPs-based approach.
The power relationship between Nv and Nnv is:
derivative: 0.8353974035228025
IsoFlops: 0.4165 / 0.5 = 0.833
'''
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.special import huber
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

import numpy as np
from utils import relative_mse, Nnv_to_d
from tqdm import tqdm
from utils import embed_dim_dict, model_size_dict, max_D_dict, steps_for_1epoch_dict
from pathlib import Path
import math


def dF_dV(V, a, b, c, d, Nnv, H):
    logv = np.log(np.minimum(V, 200_000))
    if V < 200_000:
        term1 = d * (a * logv**2 + b * logv + c)
        term2 = (Nnv + d * V) * (1 / V) * (2 * a * logv + b)
        return 6 * H * (term1 + term2)
    else:
        return 6 * H * d * (a * logv**2 + b * logv + c)

Nv_solutions = []
V_solutions = []
Nnvs_solutions = [33_000_000,
                  85_000_000,
                  151_000_000,
                  302_000_000,
                  631_000_000,
                  1130_000_000,
                  2870_000_000]

for Nnv in Nnvs_solutions:
    d = Nnv_to_d(Nnv)
    initial_guess = 1000
    V_solution = fsolve(dF_dV, initial_guess, args=(0.00639222, -0.15811069, 1.20470122, d, Nnv, 1))[0]
    print(f'Nnv={Nnv:.1e}, d={d}, searched_Vopt={int(V_solution)}')
    Nv_solutions.append(V_solution*d)

# calculcate dNv and dNnv
dNv = []
dNnv = []
for i in range(1, len(Nv_solutions)):
    dNv.append(Nv_solutions[i] / Nv_solutions[0])
    dNnv.append(Nnvs_solutions[i] / Nnvs_solutions[0])

def LSE(params, dNnv):
    gamma = params
    return gamma * np.log(dNnv)

def objective_function_Nnv(params, delta=0.001):
    prediction = LSE(params, dNnv)
    residuals = (prediction - np.log(dNv))
    return np.sum(huber(delta, residuals))


best_mse = float('inf')
best_r2 = 0
best_mse_init_guess, best_r2_init_guess = None, None
best_mse_guess, best_r2_guess = None, None
best_data_predicted = None
cnt = 0
for init_gamma in np.linspace(0, 1, 20):
    cnt += 1
    if cnt % 500 == 0:
        print('The number of init guess: ',cnt)
    initial_guess = [init_gamma]

    result = minimize(objective_function_Nnv, initial_guess,  method='L-BFGS-B')
    data_actual = np.log(dNv)
    data_predicted = np.array(LSE(result.x, dNnv))
    
    mse = relative_mse(data_actual, data_predicted)
    r2 = r2_score(data_actual, data_predicted)

    cond =  result.x[0] < 1
    if mse < best_mse and cond:
        best_mse = mse
        best_mse_init_guess = initial_guess
        best_mse_guess = result.x
        best_data_predicted = data_predicted

    if r2 > best_r2 and cond:
        best_r2 = r2
        best_r2_init_guess = initial_guess
        best_r2_guess = result.x

print(f"MSE (good MSE near to 0): {best_mse}\n\
    best_mse_init_guess is {best_mse_init_guess}\n\
    best_mse_guess is {best_mse_guess}\n\
    best_r2 (good r2 near to 1): {best_r2}\n\
    best_r2_guess is {best_r2_guess}\n\
    "
)

best_gamma = best_mse_guess[0]
print("best_gamma: ", best_gamma)



def interpolate_loss(known_flops, known_losses, target_flops):
    """
    Interpolates the expected loss for a given FLOPS value based on known FLOPS and loss pairs
    using quadratic interpolation with `interp1d` from SciPy.
    
    Args:
        known_flops (list): List of known FLOPS values.
        known_losses (list): List of corresponding known loss values.
        target_flops (float or list): The FLOPS value(s) for which to interpolate the loss.
        
    Returns:
        float or np.ndarray: The interpolated loss value(s) for the target FLOPS.
    """
    # Convert input lists to numpy arrays
    known_flops = np.array(known_flops)
    known_losses = np.array(known_losses)
    
    # Create the interpolation function
    interp_func = interp1d(known_flops, known_losses, kind='quadratic', fill_value='extrapolate')
    
    # Interpolate the loss for the target FLOPS
    interpolated_loss = interp_func(target_flops)
    return interpolated_loss

# first read the exp folder and find the optimal vocabulary size for the smallest model
# then use the derivative-based approach to predict the optimal vocabulary size for the larger models
ckpt_dir = Path('exp_data')
flops_budget = 2.7e+17
best_vocab, best_lossu = 0, 0

for exp in tqdm(sorted(ckpt_dir.glob(f'tiny_LLaMA_0000050M-*'))):
    step_cnt = 0
    num_ckpt_recode = len(list(exp.glob('*ckpt.txt')))
    
    expname = exp.name
    model_size_name = expname.split('-')[0].split('_')[-1].lstrip('0')
    d = embed_dim_dict[model_size_name]
    N = model_size_dict[model_size_name]
    V = float(expname.split('-')[1].split('_')[0][1:].replace('IsoFLOP',''))

    cur_flops, cur_lossu = [], []
    for idx,step_recode in enumerate(sorted(exp.glob('*ckpt.txt'))):
        if not step_recode.is_file():
            continue
        step = float(step_recode.name.split('-')[1])
        step_cnt += 1

        steps_for_1epoch = steps_for_1epoch_dict[model_size_name]
        D = max_D_dict[model_size_name] * (step/steps_for_1epoch) 
        Nnv = N - 2*16384*d
        flops = 6*(Nnv + V*d)*D
        cur_flops.append(flops)
        with open(step_recode) as f:
            loss_u = math.log(float(f.read()))
            cur_lossu.append(loss_u)
    
    # using existing flops and loss to interpolate the expected lossu for the given flops
    expected_lossu = interpolate_loss(cur_flops, cur_lossu, flops_budget)
    if expected_lossu < best_lossu:
        best_lossu = expected_lossu
        best_vocab_para = V * d

# find the optimal vocabulary in similar FLOPs scale in cur_flops
print(f"The best vocab parameters for the 33M model is: {best_vocab_para / 1_000_000}M")

for test_Nnv in [2.87*10**9, 3*10**9, 7*10**9, 13*10**9, 30*10**9, 70*10**9, 130*10**9, 300*10**9]:
    d = Nnv_to_d(test_Nnv)
    # predict the optimal vocabulary size
    derivative_vocab = int(best_vocab_para * (test_Nnv / 33_000_000) ** best_gamma / d)
    Nv = derivative_vocab*d
    print(f'Approach2: Nnv={test_Nnv:.1e}, Vopt-derivative:{derivative_vocab}, Nv-derivative:{Nv/10**9}B')
