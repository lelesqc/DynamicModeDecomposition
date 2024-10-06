import torch as pt
import pytest
import sys
import os
from DMD.data_processor import process_data
from DMD.functions import find_optimal_rank

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_finder_opt_rank_invalid_thr():
    """
    Test that raises an error if thr parameter is not valid.
    Since it is a percentage, it should be greater than 0 and less or equal than 100.

    """
    s = pt.tensor([5.0, 3.0, 2.0, 1.0])
    
    with pytest.raises(ValueError, match="Threshold must be positive"):
        find_optimal_rank(s, -3)
    
    with pytest.raises(ValueError, match="Threshold must be less or equal than 100"):
        find_optimal_rank(s, 200) 

def test_find_optimal_rank():
    """
    Test that verifies correct optimal rank are computer through dummy singular values.

    """
    s = pt.tensor([5.0, 3.0, 2.0, 1.0])  
    
    assert find_optimal_rank(s, 99) == 3  
    assert find_optimal_rank(s, 80) == 2 
    assert find_optimal_rank(s, 60) == 1  
