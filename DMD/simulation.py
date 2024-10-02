import logging
import torch as pt
from numpy import pi
from functions import find_optimal_rank
from flowtorch.analysis import SVD
from data_loader import load_data
from data_processor import process_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_DMD():
    _, t_steps, dt, data_matrix = process_data() 
    
    # Matrices X and X' won't be defined, slicing on data_matrix will be used instead
    rank = min(data_matrix[:, :-1].size())
    logger.info(f"The rank of matrix A is: {rank}\n")

    logger.info("Computing Singular Value Decomposition of data matrix X...\n")
  
    # In truncated SVD, we keep the greatest r = rank (of 'data_matrix') singular values
    U, s, Vh = pt.linalg.svd(data_matrix[:, :-1], full_matrices=False)
    
    # To further reduce the computational effort, we keep a certain % of singular values contribution 
    thr = 99.5
    optimal_rank = find_optimal_rank(s, thr)
    logger.info(f"The optimal rank to keep {thr}% of the singular values contribution is {optimal_rank}")
    logger.info(f"We discarded the {rank - optimal_rank} smallest singular values\n")

    # Subscript "r" represents reduced quantities   
    Ur = U[:, :optimal_rank].to(data_matrix.dtype)
    sr = s[:optimal_rank].to(data_matrix.dtype)
    Vr = Vh[:optimal_rank, :].to(data_matrix.dtype)
    
    logger.info("Proceeding with Dynamic Mode Decomposition, seek of DMD modes...\n")   
    
    sr_inv = pt.diag(1.0 / sr)    
    At = Ur.conj().T @ data_matrix[:,1:] @ Vr.conj().T @ sr_inv    # Reduced linear operator    
    eig_val, eig_vec = pt.linalg.eig(At)
    
    phi = data_matrix[:, 1:] @ Vr.conj().T @ sr_inv @ eig_vec

    logger.info(f"{phi.size(1)} modes have been collected.\n")
        
    for i in range(eig_val.size(0)):
        logger.info(f"Frequency of mode {i} is {round(pt.log(eig_val[i]).imag.item() / (2.0 * pi * dt), 2)} Hz")

    logger.info("Reconstructing data through DMD modes and computing the error...\n")

    b = pt.linalg.pinv(phi) @ data_matrix[:, 0]    # b = (phi)^-1 * x_0
    vander_matrix = pt.vander(eig_val, len(t_steps), increasing = True)
    dynamics = pt.diag(b) @ vander_matrix
    reconstruction = phi @ dynamics

    reconstruction_error = (data_matrix - reconstruction) ** 2
    mse = reconstruction_error.mean(axis = 0)

    logger.info("Reconstruction completed. \n")

    return optimal_rank, eig_val, eig_vec, phi, dynamics, reconstruction, mse
