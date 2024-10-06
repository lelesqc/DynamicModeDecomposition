import matplotlib
matplotlib.use('Agg')

import torch as pt
import matplotlib.pyplot as plt
import pytest
import sys
import os
from DMD.data_loader import load_data
from DMD.data_processor import process_data
from DMD.plotter import Plotter
from DMD.simulation import run_DMD

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def plotter():
    _, pts, _ = load_data()
    mask, _, _, _ = process_data()
    return Plotter(pts, mask)

@pytest.fixture
def data_fixture():
    _, _, _, _, _, reconstruction, _ = run_DMD()
    _, t_steps, _, data_matrix = process_data()
    return data_matrix, reconstruction, t_steps

def test_plot_data_invalid_data_length(plotter):
    """
    Test that verifies the correct raise of an error if data size doesn't match x-axis points.
    
    """    
    data = pt.tensor([1.0])

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="Size of data must match the number of points on plot's axes."):
        plotter.plot_data(ax, data, "Title")        

def test_plot_DMD_modes_empty_indices(plotter):
    """
    Test that verifies the correct raise of an error if no indices are provided.

    """    
    phi = pt.rand(10, 5)
    mode_indices = []

    with pytest.raises(ValueError, match="At least one mode index must be provided."):
        plotter.plot_DMD_modes(phi, mode_indices)

def test_plot_DMD_modes_invalid_indices(plotter):
    """
    Test that verifies the correct raise of an error if too many indices are provided.
    The maximum number of indices, namely modes available, are given by the 2nd dimension of tensor `phi`.
    
    """
    phi = pt.rand(10, 5)
    mode_indices = [0, 1, 6]  # phi.size(1) is 5, so 6 is out of bound

    with pytest.raises(IndexError, match=f"Index or indices out of bound. There are {phi.size(1)} modes that can be accessed"):
        plotter.plot_DMD_modes(phi, mode_indices)        

def test_plot_DMD_modes_single_index(plotter):
    """
    Test that verifies the correct plotting in case of a single index provided.
    Raises an exception in case of unexpected behaviour.

    """    
    _, _, _, phi, _, _, _ = run_DMD()
    mode_indices = [2]
    
    plotter.plot_DMD_modes(phi, mode_indices)        

def test_data_reconstruction_empty_times(plotter, data_fixture):
    """
    Test that verifies the correct raise of an error if `t_idx` is empty

    """    
    data_matrix = pt.rand(10, 10) 
    reconstruction = pt.rand(10, 10)
    _, _, t_steps = data_fixture
    t_idx = []

    with pytest.raises(ValueError, match="`t_idx` must contain at least one time-step to plot."):
        plotter.data_reconstruction(data_matrix, reconstruction, t_idx, t_steps)

def test_data_reconstruction_wrong_shapes(plotter, data_fixture):
    """
    Test that verifies the correct raise of an error if `data_matrix` and `reconstruction` have different sizes.

    """
    data_matrix = pt.rand(10, 10)  
    reconstruction = pt.rand(8, 10)
    _, _, t_steps = data_fixture
    times = [0, 1, 2]

    with pytest.raises(ValueError, match="`reconstruction` and `data_matrix` must have the same shape."):
        plotter.data_reconstruction(data_matrix, reconstruction, times, t_steps)

def test_data_reconstruction_single_time(plotter, data_fixture):
    """
    That that verifies the correct plotting in case of single time step provided.
    Raises an exception in case of unexpected behaviour.

    """
    data_matrix, reconstruction, t_steps = data_fixture
    data_matrix, reconstruction = data_matrix.real, reconstruction.real
    times = 3

    try:
        plotter.data_reconstruction(data_matrix, reconstruction, times, t_steps)
    except Exception as e:
        pytest.fail(f"Function raised an expected exception: {e}")

def test_reconstruction_error_wrong_shapes(plotter):
    """
    Test that verifies the correct raise of an error if `MSE_dmd` and `time_steps` have different sizes.

    """
    time_steps = list(range(10))
    mse_dmd = pt.rand(8)

    with pytest.raises(ValueError, match="`mse_dmd` must be of the same size as `time_steps`."):
        plotter.reconstruction_error(time_steps, mse_dmd)
