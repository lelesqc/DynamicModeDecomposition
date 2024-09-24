import pytest
import torch as pt
from unittest.mock import MagicMock
from data_loader import load_data
from flowtorch.data import FOAMDataloader

# ------------------  FIXTURES  -----------------------

# FOAMDataloader attributes (write_times, field_names, vertices) cannot be assigned directly.
# We have to make use of mocking

@pytest.fixture
def empty_times_loader():
    """
    Fixture that simulates empty times using mocking.
    
    """
    mock_loader = MagicMock()
    mock_loader.write_times = [] 
    mock_loader.field_names = {'4.0': 'velocity'}
    mock_loader.vertices = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
    return mock_loader

@pytest.fixture
def empty_fields_loader():
    """
    Fixture that simulates empty fields using mocking.
    
    """
    mock_loader = MagicMock()
    mock_loader.write_times = ["1.0", "2.0", "3.0"]
    mock_loader.field_names = {}
    mock_loader.vertices = pt.tensor([[1.0, 2.0], [3.0, 4.0]])
    return mock_loader

@pytest.fixture
def nan_pts_loader():
    """
    Fixture that simulates NaN pts using mocking.
    
    """
    mock_loader = MagicMock()
    mock_loader.write_times = ["1.0", "2.0", "3.0"]
    mock_loader.field_names = {'4.0': 'velocity'}
    mock_loader.vertices = pt.Tensor([[1.0, 2.0], [float('nan'), 1.0]])
    return mock_loader

@pytest.fixture
def empty_pts_loader():
    """
    Fixture that simulates empty pts using mocking.
    
    """
    mock_loader = MagicMock()
    mock_loader.write_times = ["1.0", "2.0", "3.0"] 
    mock_loader.field_names = {'4.0': 'velocity'}
    mock_loader.vertices = pt.Tensor([])
    return mock_loader


# ---------------------  TESTS  ---------------------------

def test_load_data_empty_times(empty_times_loader):
    """
    Test that verifies the correct raise of an error if times is empty.
    It is paired with empty_times_loader fixture.

    """
    with pytest.raises(ValueError, match="Time steps are empty."):
        load_data(loader=empty_times_loader)

def test_load_data_empty_fields(empty_fields_loader):
    """
    Test that verifies the correct raise of an error if fields is empty.
    It is paired with empty_fields_loader fixture.

    """
    with pytest.raises(ValueError, match="No fields available in the dataset."):
        load_data(loader=empty_fields_loader)

def test_load_data_nan_pts(nan_pts_loader):
    """
    Test that verifies the correct raise of an error if pts contains NaN values.
    It is paired with nan_pts_loader fixture.

    """
    with pytest.raises(ValueError, match="One or more vertices value is NaN."):
        load_data(loader=nan_pts_loader)

def test_load_data_empty_pts(empty_pts_loader):
    """
    Test that verifies the correct raise of an error if pts is empty.
    It is paired with empty_pts_loader fixture.

    """
    with pytest.raises(ValueError, match="Vertices are empty."):
        load_data(loader=empty_pts_loader)
    
def test_apply_mask_valid():
    """
    Test that verifies mask is correctly implemented based on pts.
    It must have the same size of the tensor on which it has been applied.
    
    """
    _, _, pts, mask = load_data()    
    
    assert mask.size(0) > 0, "Mask is empty"
    assert mask.size(0) == pts.size(0), "Mask has different size with respect to pts"



