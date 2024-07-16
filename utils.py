import numpy as np
import math
from scipy.interpolate import griddata, interp1d


# The max training tokens (D) used for each model family
max_D_list = [1.2*10**9,3.0*10**9, 5.3*10**9, 11.7*10**9, 27.7*10**9, 54.8*10**9]



def D_to_H(D,V=16384):
    logv = np.log(np.minimum(V, 200_000))
    term = 0.00639222*logv**2 - 0.15811069*logv + 1.20470122
    return D/term

def H_to_D(H,V=16384):
    logv = np.log(np.minimum(V, 200_000))
    term = 0.00639222*logv**2 - 0.15811069*logv + 1.20470122
    return H*term


def generate_interpolation_log(values, num=8, var=0):
    return np.logspace(np.log10(values.min()*(1-var)), np.log10(values.max()*(1+var)), num)

def generate_interpolation_linear(values, num=8, var=0):
    return np.linspace(values.min()*(1-var), values.max()*(1+var), num)

def Nnv_to_d(Nnv):
    if Nnv <= 50_000_000:
        d = 512
    elif 50_000_000 < Nnv <= 200_000_000:
        d = 768 
    elif 200_000_000 < Nnv <= 500_000_000:
        d = 1024 
    elif 500_000_000 < Nnv <= 1_000_000_000:
        d = 1536     
    elif 1_000_000_000 < Nnv <= 2_000_000_000:
        d = 2048 
    elif 2_000_000_000 < Nnv <= 5_000_000_000:
        d = 3200 
    elif 5_000_000_000 < Nnv <= 10_000_000_000:
        d = 4096 
    elif 10_000_000_000 < Nnv <= 20_000_000_000:
        d = 5120    
    elif 20_000_000_000 < Nnv <= 50_000_000_000:
        d = 6048  
    elif 50_000_000_000 < Nnv <= 100_000_000_000:
        d = 8192  
    elif 100_000_000_000 < Nnv <= 200_000_000_000:
        d = 12288  
    elif 200_000_000_000 < Nnv <= 500_000_000_000:
        d = 16384   
    elif 500_000_000_000 < Nnv <= 1000_000_000_000:
        d = 20480
    else:
        d = 24576
        # raise ValueError()       
    return float(d) 


def func_flops(Nnv, H, V):
    d = Nnv_to_d(Nnv)                    
    logv = math.log(min(V, 200_000))
    return 6*(Nnv+V*d)*H*(0.00639222*logv**2 - 0.15811069*logv + 1.20470122)

def interpolate(Nnv_data,H_data, V_data,flops_data, L_values, num_model, num_v,num_eval):
    reshape_Nnv = np.reshape(Nnv_data, (num_model, num_v,num_eval ))
    reshape_H = np.reshape(H_data, (num_model, num_v,num_eval ))
    reshape_V = np.reshape(V_data, (num_model, num_v,num_eval ))
    reshape_L = np.reshape(L_values, (num_model, num_v,num_eval))

    interpolated_Nnv, interpolated_H, interpolated_V = [],[],[]
    interpolated_flops, interpolated_loss = [], []

    new_Nnv = generate_interpolation_log(np.unique(Nnv_data), 50)
    new_H = generate_interpolation_log(np.unique(H_data), 50)
    for vid in range(num_v):
        cur_V = reshape_V[0,vid,0]
        cur_all_Nnv = reshape_Nnv[:,vid,:].ravel()
        cur_all_H = reshape_H[:,vid,:].ravel()
        cur_all_L = reshape_L[:,vid,:].ravel()

        new_points = np.array(list(zip(new_Nnv, new_H)))
        points = np.array(list(zip(cur_all_Nnv, cur_all_H)))
        new_L = griddata(points, cur_all_L, new_points, method='cubic')
        for nnv,h,l in zip(new_Nnv, new_H, new_L):
            if np.isnan(l):
                continue
            f = func_flops(nnv, h, cur_V)
            interpolated_Nnv.append(nnv)
            interpolated_H.append(h)
            interpolated_V.append(cur_V)
            interpolated_flops.append(f)
            interpolated_loss.append(l)    

    new_V = generate_interpolation_log(np.unique(V_data), 20)
    for modelid in range(num_model):
        cur_Nnv = reshape_Nnv[modelid,0,0]
        cur_all_V = reshape_V[modelid,:,0]
        cur_all_L = reshape_L[modelid,:,-1]
        interpolation_function = interp1d(cur_all_V, cur_all_L, kind='quadratic', fill_value='extrapolate')
        new_L = interpolation_function(new_V)
        new_H = np.array([D_to_H(max_D_list[modelid], V=i) for i in new_V])
        for v,h,l in zip(new_V, new_H,new_L):
            if np.isnan(l):
                continue
            f = func_flops(cur_Nnv, h, v)
            interpolated_Nnv.append(cur_Nnv)
            interpolated_H.append(h)
            interpolated_V.append(v)
            interpolated_flops.append(f)
            interpolated_loss.append(l)            
    interpolated_Nnv, interpolated_H, interpolated_V, interpolated_flops, interpolated_loss = \
        np.array(interpolated_Nnv), np.array(interpolated_H), np.array(interpolated_V), np.array(interpolated_flops), np.array(interpolated_loss)
    return  interpolated_Nnv, interpolated_H, interpolated_V, interpolated_flops, interpolated_loss           


def merge_nearest_flops(flops_data, num_bin=10):
    '''
    flops_data is an ascending sequence.
    Given a argument num_bin, we divide the values in flops_data into num_bin from minimum to maximum.
    '''
    bins_log = np.logspace(np.log10(flops_data.min()), np.log10(flops_data.max()), num_bin+1)[1:]
    indices = np.digitize(flops_data, bins_log) # - 1
    flops_data_binned = np.zeros_like(flops_data)
    for i in range(num_bin):
        bin_mask = indices == i
        flops_data_binned[bin_mask] = flops_data[bin_mask].mean()
    return flops_data_binned


def relative_mse(actual, predicted):
    errors = actual - predicted
    mse = np.mean(np.square(errors))
    mean_squared = np.mean(actual)**2
    relative_mse_mean_squared = mse / mean_squared
    return relative_mse_mean_squared


def remove_outlier(flops, Nnvopt, Nvopt, Hopt, best_K_set,best_alpha_set):
    flops,  Nnvopt, Nvopt, Hopt = np.array(flops), np.array(Nnvopt), np.array(Nvopt), np.array(Hopt)
    kept_idx = []
    for idx,f in enumerate(flops):
        ypred0 = np.exp(best_K_set[0])*(f)**best_alpha_set[0]
        ypred1 = np.exp(best_K_set[1])*(f)**best_alpha_set[1]
        ypred2 = np.exp(best_K_set[2])*(f)**best_alpha_set[2]
        if abs(Nnvopt[idx]- ypred0)/ ypred0 < 0.4 and \
            abs(Nvopt[idx]- ypred1)/ ypred1 < 0.4 and \
            abs(Hopt[idx]- ypred2)/ ypred2 < 0.4 :
            kept_idx.append(idx)
    kept_idx = np.array(kept_idx)
    return flops[kept_idx], Nnvopt[kept_idx], Nvopt[kept_idx], Hopt[kept_idx]    