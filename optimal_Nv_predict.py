from scipy.optimize import fsolve
import numpy as np
from utils import Nnv_to_d
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adjustText import adjust_text

def Nnvopt_to_flops(Nnv):
    '''Return the corresponding training-optimal FLOPs budget
     given the non-vocabulary parameters Nnv'''
    FLOPs = ( Nnv/np.exp(-2.4846510161625193)) ** (1/0.5)
    return FLOPs


def flops_to_Nnvopt(FLOPs):
    '''Return the corresponding training-optimal non-vocabulary parameters Nnv
     given the FLOPs budget'''
    return np.exp(-2.4846510161625193) * FLOPs **0.5


def approach1_isoflops(Nnv):
    '''Predict the training-optimal vocabulary parameters by the approach 1:
    Build the relationship between studied attributes and FLOPs'''    
    d = Nnv_to_d(Nnv)
    FLOPs = ( Nnv/np.exp(-2.4846510161625193)) ** (1/0.5)
    Nv = np.exp(-1.589031299255507)* FLOPs ** 0.4163622634135234
    return int(Nv)

def approach2_derivative(Nnv):
    '''Predict the training-optimal vocabulary parameters by the approach 2:
    Derivative-based fast estimation'''       
    d = Nnv_to_d(Nnv)
    best_vocab_para = 3145728
    best_alpha = 0.8353974035228025
    return int(best_vocab_para * (Nnv / 33_000_000) ** best_alpha)

def approach3_isoloss(Nnv, FLOPs=None):   
    '''Predict the training-optimal vocabulary parameters by the approach 3:
    Parametric fit of loss function.
    Different from the approach 1 & 2 that assumes the the training data and 
    non-vocabulary parameters are EQUALLY scaled to essure the optimal compute allocation,
    the approach 3 is more flexible that it can also be used in the cases the training data is
    not EQUALLY scaled with the non-vocabulary parameters, for example, the number of data 
    is insufficient or overly sufficient. One can assign a FLOPs budget to 
    adjust the number of available training data.
     '''       
    def dl_dv(V, Nnv, d, F):
        term1 = 0  # Derivative of -E
        term2 = 0  # Derivative of A1/[Nnv]^alpha1
        term3 = -alpha2 * A2 * d / (V * d) ** (alpha2 + 1)
        u = F / (6 * (Nnv + V * d))
        du_dV = F * d / (6 * (Nnv + V * d) ** 2)
        term4 = beta * B * du_dV / (u ** (beta + 1))
        return term1 + term2 + term3 + term4
    A1, A2, B, E = 1.8313851559554126, 0.19584238398665638, 2.1241123120064955, 5.5327846803337435,
    alpha1, alpha2, beta = 0.44660634152009615, 0.6707374679896795, 0.44660634152009615
    
    d = Nnv_to_d(Nnv)
    if FLOPs is None:
        FLOPs = Nnvopt_to_flops(Nnv)
    # normalization
    Nnv = Nnv / 1_000_000
    d = d / 1_000   
    FLOPs = FLOPs / (1_000_000_000*1_000_000)
    V = fsolve(dl_dv, 1, args=(Nnv,d,FLOPs))[0]
    # de-normalization
    Nnv = Nnv * 1_000_000
    d = d * 1_000   
    FLOPs = FLOPs * (1_000_000_000*1_000_000)
    return int(V*1000*d)    


if __name__ == '__main__':
    '''
    By using the coefficient fitted in the proposed 3 approaches, this code
    provide an example about how to predict the optimal vocabulary 
    parameters (Nv) and vocabulary size, given the non-vocabulary parameters (Nnv).
    '''
    Nnv = 7*10**9
    Nvopt_app1 = approach1_isoflops(Nnv)
    Nvopt_app2 = approach2_derivative(Nnv)
    Nvopt_app3 = approach3_isoloss(Nnv)  
    d = Nnv_to_d(Nnv)
    Vopt_app1, Vopt_app2, Vopt_app3 = int(Nvopt_app1/d), int(Nvopt_app2/d), int(Nvopt_app3/d)
    print(f'Given Nnv={Nnv}: The predicted optimal vocabulary size is {Nvopt_app1}, {Nvopt_app2}, {Nvopt_app3} by the 3 proposed approaches.\
    The predicted optimal vocabulary size is {Vopt_app1}, {Vopt_app2}, {Vopt_app3} by the 3 proposed approaches.')  
    