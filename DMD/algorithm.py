import logging
import torch as pt
import numpy as np
from flowtorch.analysis import SVD
from functions import find_optimal_rank
from data_processor import process_data

logger = logging.getLogger(__name__)

def run_DMD():
    _, t_steps, dt, data_matrix = process_data() 
    if data_matrix.dtype not in (pt.complex64, pt.complex128):
        data_matrix = data_matrix.type(pt.cfloat)
    
    # Matrices X and X' won't be defined, slicing on data_matrix will be used instead
    rank = min(data_matrix[:, :-1].size())
    print("The rank of matrix A is:", rank)

    logger.info("Computing Singular Value Decomposition of data matrix X...")
  
    # In truncated SVD, we keep the greatest r = rank (of 'data_matrix') singular values
    U, s, Vh = pt.linalg.svd(data_matrix[:, :-1], full_matrices=False)
    
    # To further reduce the computational effort, we keep a certain % of singular values contribution 
    thr = 99.5
    optimal_rank = find_optimal_rank(s, thr)
    print("The optimal rank to keep", thr, "of the singular values contribution is", optimal_rank)
    print("We discarded the", rank - optimal_rank, "smallest singular values")

    # Subscript "r" represents reduced quantities   
    Ur = U[:, :optimal_rank].to(data_matrix.dtype)
    sr = s[:optimal_rank].to(data_matrix.dtype)
    Vr = Vh[:optimal_rank, :].to(data_matrix.dtype)
    
    logger.info("Proceeding with Dynamic Mode Decomposition, seek of DMD modes...")   
    
    sr_inv = pt.diag(1.0 / sr)    
    At = Ur.conj().T @ data_matrix[:,1:] @ Vr.conj().T @ sr_inv    # Reduced linear operator    
    eig_val, eig_vec = pt.linalg.eig(At)
    
    phi = data_matrix[:, 1:] @ Vr.conj().T @ sr_inv @ eig_vec
        
    for i in range(eig_val.size(0)):
        freq = round(float(pt.log(eig_val[i]).imag/(2.0 * np.pi * dt)), 3)
        print(f"Frequency of mode", i, "is", freq, "Hz")

    logger.info("Reconstructing data through DMD modes and computing the error...")

    b = pt.linalg.pinv(phi) @ data_matrix[:, 0]    # b = (phi)^-1 * x_0
    vander_matrix = pt.vander(eig_val, len(t_steps), increasing = True)
    dynamics = pt.diag(b) @ vander_matrix
    reconstruction = phi @ dynamics
