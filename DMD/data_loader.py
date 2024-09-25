from numpy import isnan
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from config import DATASET_NAME, MASK_LOWER_BOUND, MASK_UPPER_BOUND

def load_data(loader=None):
    if loader is None:
        dataset = DATASETS[DATASET_NAME]
        loader = FOAMDataloader(dataset)
    
    times = loader.write_times
    fields = loader.field_names
    pts = loader.vertices

    if pts.numel() == 0:
        raise ValueError("Vertices are empty.")
    
    pts = pts[:, :2]    # Discard z-coordinate since simulation is only 2D

    if not times:
        raise ValueError("Time steps are empty.")
    
    if not fields:
        raise ValueError("No fields available in the dataset.")

    if isnan(pts).any():
        raise ValueError("One or more vertices value is NaN.") 

    return times, fields, pts, loader
