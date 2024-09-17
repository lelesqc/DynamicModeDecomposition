import torch as pt
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from config import DATASET_NAME, MASK_LOWER_BOUND, MASK_UPPER_BOUND

def load_data():
    dataset = DATASETS[DATASET_NAME]
    loader = FOAMDataloader(dataset)
    times = loader.write_times
    fields = loader.field_names
    pts = loader.vertices[:, :2]    # Discard z-coordinate since simulation is only 2D
    
    if not times or not fields or not pts.size(0):
        raise ValueError("One or more required data items are empty. Verify that data has been loaded correctly.")
    
    return loader, times, fields, pts

def apply_mask(pts):
    mask = mask_box(pts, lower=MASK_LOWER_BOUND, upper=MASK_UPPER_BOUND)
    if mask.size(0) == 0:
        raise ValueError("Mask is empty. Please check data.")
    return mask
