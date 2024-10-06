import sys
import os
from flowtorch.analysis import DMD
from DMD.simulation import run_DMD
from DMD.data_processor import process_data
from numpy import allclose
from torch import complex128

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_run_DMD_eigs():
    """
    Test that verifies eigenvalues and eigenvectors computed are as the expected.
    Comparison is made with built-in properties of flowtorch.analysis.DMD class.

    """
    
    _, eig_val, eig_vec, _, _, _, _ = run_DMD()
    _, _, dt, data_matrix = process_data()
    
    dmd = DMD(data_matrix, dt=dt, rank = eig_val.size(0))
    eig_val_dmd = dmd.eigvals
    eig_vec_dmd = dmd.eigvecs

    assert allclose(eig_val_dmd, eig_val), "Eigenvalues are different from the expected"
    assert allclose(eig_vec_dmd, eig_vec), "Eigenvectors are different from the expected"

def test_run_DMD_modes():
    """
    Test that verifies modes computed are as the expected.
    Comparison is made with built-in properties of flowtorch.analysis.DMD class.

    """
    _, _, _, phi, _, _, _ = run_DMD()
    _, _, dt, data_matrix = process_data()
    
    dmd = DMD(data_matrix, dt=dt, rank = phi.size(1))
    modes_dmd = dmd.modes

    assert allclose(modes_dmd, phi), "Modes are different from the expected"

def test_run_DMD_reconstruction():
    """
    Test that verifies computed data matrix reconstruction is as the expected.
    Comparison is made with built-in properties of flowtorch.analysis.DMD class.

    """
    _, eig_val, eig_vec, _, _, reconstruction, _ = run_DMD() 
    _, _, dt, data_matrix = process_data()
    
    dmd = DMD(data_matrix, dt=dt, rank = eig_val.size(0))
    reconstruction_dmd = dmd.reconstruction

    reconstruction = reconstruction.to(complex128)
    reconstruction_dmd = reconstruction_dmd.to(complex128)

    # Reconstruction is the result of many tensor operations, so a higher tolerance is needed for numerical precision 
    tolerance = 1e-3

    assert allclose(reconstruction_dmd, reconstruction, atol=tolerance), "Data matrix reconstruction is different from the expected"
