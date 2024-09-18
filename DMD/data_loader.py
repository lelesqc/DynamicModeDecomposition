import torch as pt
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from config import DATASET_NAME, MASK_LOWER_BOUND, MASK_UPPER_BOUND

def load_data(dataset_path=None):
    try:
        dataset = DATASETS[DATASET_NAME]
        loader = FOAMDataloader(dataset)
        times = loader.write_times
        fields = loader.field_names
        pts = loader.vertices[:, :2]    # Discard z-coordinate since simulation is only 2D
    
        if not times or not fields or not pts.size(0):
            raise ValueError("One or more required data items are empty. Verify that data has been loaded correctly.")
    
        return loader, times, fields, pts

    # Check for any other possible error occurring internally
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading data: {str(e)}")


def apply_mask(pts):
    try:
        mask = mask_box(pts, lower=MASK_LOWER_BOUND, upper=MASK_UPPER_BOUND)        
        if mask.size(0) == 0:
            raise ValueError("Mask is empty. Please check data loading.")
            
        return mask

    # Check for any other possible error occurring internally
    except Exception as e:
        raise RuntimeError(f"An error occurred while applying the mask: {str(e)}")
