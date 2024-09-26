import pytest
import torch as pt
from data_loader import load_data
from data_processor import process_data
from data_processor import process_data

def test_apply_mask_valid():
    """
    Test that verifies mask is correctly implemented based on pts.
    It must have the same size of the tensor on which it has been applied.
    
    """
    _, pts, _ = load_data()   
    mask, _, _, _ = process_data()
    
    assert mask.size(0) > 0, "Mask is empty"
    assert mask.size(0) == pts.size(0), "Mask has different size with respect to pts"

def test_t_steps_valid():
    """
    Test that verifies t_steps still contains values after filtering.
    
    """
    _, t_steps, _, _ = process_data()

    assert len(t_steps) > 0, "t_steps is empty"   

def test_dt_correct():
    """
    Test that verifies dt is positive and equal to the expected.
    
    """
    _, _, dt, _ = process_data()

    known_times = ["9.0", "9.025"]
    expected_dt = round(float(known_times[1]) - float(known_times[0]), 3)
    
    assert dt == expected_dt, "dt doesn't correspond to the value expected"

def test_data_matrix_shape():
    """
    Test that verifies data_matrix has the correct shape.
    
    """
    mask, t_steps, _, data_matrix = process_data()
    
    num_masked_points = pt.count_nonzero(mask)
    num_time_steps = len(t_steps)
    
    assert data_matrix.shape == (num_masked_points, num_time_steps), (
        f"data_matrix has shape {data_matrix.shape}, "
        f"but it should have shape ({num_masked_points}, {num_time_steps})."
    )

    assert not pt.isnan(data_matrix).any(), "data_matrix contains NaN values"
