from numpy import isnan
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from DMD.config import DATASET_NAME

def load_data(loader=None):
    """
    Function that loads data of the chosen dataset and verifies their integrity.

    Parameters:
        loader (FOAMDataloader, optional): FOAMDataloader object. It is set different from None in tests, otherwise defined as the loader of the default dataset.

    Returns:
        times (list): List of time steps available as strings.
        pts (torch.FloatTensor): Vertices of the grid.
        loader (FOAMDataloader): Chosen FOAMDataloader object.

    Raises:
        ValueError: If `pts` is empty
        ValueError: If `times` is empty
        ValueError: If `pts` tensor contains NaN values.

    """
    
    if loader is None:
        dataset = DATASETS[DATASET_NAME]
        loader = FOAMDataloader(dataset)
    
    times = loader.write_times
    pts = loader.vertices

    if pts.numel() == 0:
        raise ValueError("Vertices are empty.")
    
    pts = pts[:, :2]    # Discard z-coordinate since simulation is only 2D

    if not times:
        raise ValueError("Time steps are empty.")

    if isnan(pts).any():
        raise ValueError("One or more vertices value is NaN.") 

    return times, pts, loader
    if isnan(pts).any():
        raise ValueError("One or more vertices value is NaN.") 

    return times, pts, loader
