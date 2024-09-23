import numpy as np
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from config import DATASET_NAME, MASK_LOWER_BOUND, MASK_UPPER_BOUND

def load_data():
    dataset = DATASETS[DATASET_NAME]
    loader = FOAMDataloader(dataset)
    times = loader.write_times
    fields = loader.field_names
    pts = loader.vertices[:, :2]    # Discard z-coordinate since simulation is only 2D

    if not times or np.isnan(times).any():
        raise ValueError("Error: Time steps are either empty or contain NaN values.")
    
    if not fields:
        raise ValueError("Error: No fields available in the dataset.")
    
    if pts.size == 0 or np.isnan(pts).any():
        raise ValueError("Error: Vertices are either empty or contain NaN values.")

    mask = mask_box(pts, lower=MASK_LOWER_BOUND, upper=MASK_UPPER_BOUND)

    return times, fields, pts, mask
